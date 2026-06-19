import Levenshtein
from typing import List, Tuple, Dict
import numpy as np
from fastdtw import fastdtw
import librosa
import torch
import torchaudio.functional as F

class PronunciationScorer:
    def __init__(self):
        self.weights = {
            'phoneme': 0.5,
            'duration': 0.2,
            'stress': 0.2,
            'pitch': 0.1
        }
    
    def _get_alignment_ops(self, pred: List[str], ref: List[str]) -> List[Tuple[str, str]]:
        """
        Returns aligned phoneme pairs with gaps marked as '-' 
        using Levenshtein edit operations
        """
        aligned = []
        i, j = 0, 0
        ops = Levenshtein.editops(ref, pred)
        
        for op in ops:
            # Add matching phonemes before this edit
            while i < op[1] and j < op[2]:
                aligned.append((pred[j], ref[i]))
                i += 1
                j += 1
                
            # Handle the edit operation
            if op[0] == 'replace':
                aligned.append((pred[op[2]], ref[op[1]]))
                i += 1
                j += 1
            elif op[0] == 'delete':
                aligned.append(('-', ref[op[1]]))
                i += 1
            elif op[0] == 'insert':
                aligned.append((pred[op[2]], '-'))
                j += 1
        
        # Add remaining matching phonemes
        while i < len(ref) and j < len(pred):
            aligned.append((pred[j], ref[i]))
            i += 1
            j += 1
            
        return aligned
    
    def phoneme_accuracy(self, pred: List[str], ref: List[str]) -> Tuple[float, List[Tuple[str, str]]]:
        """
        Returns:
        - accuracy score (0-1)
        - aligned phoneme pairs with gaps
        """
        aligned = self._get_alignment_ops(pred, ref)
        correct = sum(1 for p, r in aligned if p == r)
        total_ref = len([r for _, r in aligned if r != '-'])
        return (correct / total_ref) if total_ref > 0 else 0.0, aligned
    
    def get_error_stats(self, aligned: List[Tuple[str, str]]) -> Dict[str, int]:
        """Returns counts of substitutions, insertions, deletions"""
        stats = {'sub': 0, 'ins': 0, 'del': 0}
        for p, r in aligned:
            if p == '-' and r != '-':
                stats['del'] += 1
            elif p != '-' and r == '-':
                stats['ins'] += 1
            elif p != r:
                stats['sub'] += 1
        return stats
    
    def duration_score(self, 
                     pred_times: List[Tuple[float, float]], 
                     ref_times: List[Tuple[float, float]],
                     aligned_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Calculate duration metrics for aligned phonemes
        
        Returns:
            {
                'accuracy': 0-1 score,
                'avg_ratio': average duration ratio,
                'error_ms': average error in milliseconds
            }
        """
        if not pred_times or not ref_times:
            return {'accuracy': 0.0, 'avg_ratio': 1.0, 'error_ms': 0.0}
            
        scores = []
        ratios = []
        errors = []
        pred_idx, ref_idx = 0, 0
        
        for p_phn, r_phn in aligned_pairs:
            # Only compare when both phonemes exist
            if p_phn != '-' and r_phn != '-':
                p_start, p_end = pred_times[pred_idx]
                r_start, r_end = ref_times[ref_idx]
                p_dur = p_end - p_start
                r_dur = r_end - r_start
                
                if r_dur > 0:
                    ratio = p_dur / r_dur
                    ratios.append(ratio)
                    errors.append(abs(p_dur - r_dur) * 1000)
                    
                    # Accuracy score (1 - normalized error)
                    norm_error = min(1, abs(1 - ratio))
                    scores.append(1 - norm_error)
                
                pred_idx += 1
                ref_idx += 1
            else:
                if p_phn == '-': ref_idx += 1
                if r_phn == '-': pred_idx += 1
        
        if not scores:
            return {'accuracy': 0.0, 'avg_ratio': 1.0, 'error_ms': 0.0}
            
        return {
            'accuracy': sum(scores) / len(scores),
            'avg_ratio': sum(ratios) / len(ratios),
            'error_ms': sum(errors) / len(errors)
        }

    def _extract_pitch_contour(self, waveform, sr, phoneme_times):
        """Extract pitch using librosa's pyin algorithm"""
        pitch_contours = []
        try:
            # Ensure waveform is 1D numpy array
            if isinstance(waveform, np.ndarray):
                if waveform.ndim > 1:
                    waveform = waveform.squeeze()
            
            # Extract pitch using librosa's pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Handle NaN values
            f0 = np.nan_to_num(f0, nan=0.0)
            
            # Extract per-phoneme segments
            hop_length = 512  # librosa default
            for start, end in phoneme_times:
                start_idx = int(start * sr / hop_length)
                end_idx = int(end * sr / hop_length)
                segment = f0[start_idx:end_idx]
                # Filter out unvoiced frames (0 values)
                segment_voiced = segment[segment > 0]
                pitch_contours.append(segment_voiced)
                
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return []
        
        return pitch_contours

    def pitch_score(self, pred_waveform, ref_waveform, sr, aligned_pairs, pred_times, ref_times):
        """Compare pitch contours using DTW"""
        # Ensure waveforms are numpy arrays
        if not isinstance(pred_waveform, np.ndarray):
            pred_waveform = np.array(pred_waveform)
        if not isinstance(ref_waveform, np.ndarray):
            ref_waveform = np.array(ref_waveform)
        
        # Squeeze to 1D if needed
        pred_waveform = pred_waveform.squeeze()
        ref_waveform = ref_waveform.squeeze()
        
        pred_contours = self._extract_pitch_contour(pred_waveform, sr, pred_times)
        ref_contours = self._extract_pitch_contour(ref_waveform, sr, ref_times)
        
        metrics = {'similarity': [], 'error_hz': [], 'correlation': []}
        p_idx, r_idx = 0, 0
        
        for p_phn, r_phn in aligned_pairs:
            if p_phn != '-' and r_phn != '-':
                if p_idx >= len(pred_contours) or r_idx >= len(ref_contours):
                    break
                    
                p_contour = pred_contours[p_idx]
                r_contour = ref_contours[r_idx]
                
                if len(p_contour) > 1 and len(r_contour) > 1:
                    # Dynamic Time Warping distance
                    dtw_dist, _ = fastdtw(p_contour, r_contour)
                    metrics['similarity'].append(1 / (1 + dtw_dist))
                    
                    # Mean absolute error (Hz)
                    metrics['error_hz'].append(abs(p_contour.mean() - r_contour.mean()))
                    
                    # Pearson correlation
                    min_len = min(len(p_contour), len(r_contour))
                    if min_len > 1:
                        corr = np.corrcoef(p_contour[:min_len], r_contour[:min_len])[0, 1]
                        metrics['correlation'].append(0 if np.isnan(corr) else corr)
                
                p_idx += 1
                r_idx += 1
            else:
                if p_phn == '-': r_idx += 1
                if r_phn == '-': p_idx += 1
        
        if not metrics['similarity']:
            return {'similarity': 0.0, 'error_hz': 0.0, 'correlation': 0.0}
            
        return {
            'similarity': np.mean(metrics['similarity']),
            'error_hz': np.mean(metrics['error_hz']),
            'correlation': np.mean(metrics['correlation']) if metrics['correlation'] else 0.0
        }

    def _extract_stress(self, phonemes):
        """Extract phoneme-stress pairs from ARPAbet/IPA-style phonemes"""
        stress_pairs = []
        for phn in phonemes:
            if len(phn) > 2 and phn[-1].isdigit():
                stress_pairs.append((phn[:-1], phn[-1]))
            else:
                stress_pairs.append((phn, '0'))
        return stress_pairs

    def stress_score(self,
                   pred_phonemes: List[str],
                   ref_phonemes: List[str],
                   aligned_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Compare stress patterns between prediction and reference
        
        Returns:
            {
                'accuracy': 0-1 score,
                'error_stats': counts by error type
            }
        """
        pred_stress = self._extract_stress([p for p, _ in aligned_pairs if p != '-'])
        ref_stress = self._extract_stress([r for _, r in aligned_pairs if r != '-'])
        
        if not ref_stress:
            return {'accuracy': 0.0, 'error_stats': {}}
            
        correct = 0
        error_stats = {
            'missing_stress': 0,
            'extra_stress': 0,
            'wrong_stress': 0
        }
        
        for (p_phn, p_str), (r_phn, r_str) in zip(pred_stress, ref_stress):
            if p_str == r_str:
                correct += 1
            elif p_str == '0' and r_str != '0':
                error_stats['missing_stress'] += 1
            elif p_str != '0' and r_str == '0':
                error_stats['extra_stress'] += 1
            else:
                error_stats['wrong_stress'] += 1
                
        return {
            'accuracy': correct / len(ref_stress),
            'error_stats': error_stats
        }
    
    def compute_scores(self, 
                      pred_phonemes: List[str], 
                      ref_phonemes: List[str],
                      pred_times: List[Tuple[float, float]] = None,
                      ref_times: List[Tuple[float, float]] = None,
                      pred_waveform = None,
                      ref_waveform = None,
                      sr: int = None) -> Dict:
        """Enhanced scoring interface with pitch analysis"""
        accuracy, aligned = self.phoneme_accuracy(pred_phonemes, ref_phonemes)
        results = {
            'phoneme': accuracy,
            'error_stats': self.get_error_stats(aligned),
            'aligned_pairs': aligned
        }
        
        if pred_times and ref_times:
            results['duration'] = self.duration_score(pred_times, ref_times, aligned)
            
        if all(x is not None for x in [pred_waveform, ref_waveform, sr]):
            results['pitch'] = self.pitch_score(
                pred_waveform, ref_waveform, sr, aligned, pred_times, ref_times)
                
        # Add stress analysis if phonemes contain stress markers
        if any(any(c.isdigit() for c in phn) for phn in ref_phonemes):
            results['stress'] = self.stress_score(pred_phonemes, ref_phonemes, aligned)
            
        return results

    def ctc_forced_align(self, log_probs: torch.Tensor, targets: torch.Tensor, blank_id: int = 0) -> List[Tuple[int, int]]:
        """
        Computes CTC forced alignment for batch_size=1.
        
        Args:
            log_probs: Tensor of shape (1, Time, Vocab)
            targets: Tensor of shape (1, Target_Len)
            blank_id: Index of blank token
            
        Returns:
            List of (start_frame, end_frame) matching each token in targets.
        """
        B, T, C = log_probs.shape
        L = targets.shape[1]
        
        print(f"[DEBUG ctc_forced_align] Shape: T={T}, L={L}, blank_id={blank_id}", flush=True)
        
        # Move inputs to CPU to avoid CUDA kernel/driver binary compatibility segfaults
        # and multi-GPU device mapping issues in torchaudio's C++ extension.
        log_probs_cpu = log_probs.cpu()
        targets_cpu = targets.cpu()
        
        targets_list = targets_cpu[0].numpy().tolist()
        print(f"[DEBUG ctc_forced_align] Target list: {targets_list}", flush=True)
        
        # Validate constraints to prevent C++ out-of-bounds/assertion crashes
        # 1. Target sequence cannot be empty
        # 2. Input frames must be >= target length
        # 3. Target sequence must not contain the blank/pad token
        if L == 0 or T < L or blank_id in targets_list:
            print(f"Warning: CTC alignment constraints violated (T={T}, L={L}, blank_in_target={blank_id in targets_list}). Falling back to linear alignment.", flush=True)
            intervals = []
            step = T / max(L, 1)
            for idx in range(L):
                s = int(idx * step)
                e = int((idx + 1) * step) - 1
                intervals.append((s, max(s, e)))
            return intervals
            
        input_lengths = torch.tensor([T], dtype=torch.long, device="cpu")
        target_lengths = torch.tensor([L], dtype=torch.long, device="cpu")
        
        # Log softmax along vocab dimension
        log_probs_norm = torch.log_softmax(log_probs_cpu, dim=-1)
        
        print(f"[DEBUG ctc_forced_align] Calling F.forced_align on CPU. log_probs_norm shape: {log_probs_norm.shape}", flush=True)
        
        try:
            # torchaudio forced_align on CPU
            alignments, scores = F.forced_align(
                log_probs_norm, 
                targets_cpu, 
                input_lengths=input_lengths, 
                target_lengths=target_lengths, 
                blank=blank_id
            )
            
            print("[DEBUG ctc_forced_align] F.forced_align completed successfully", flush=True)
            path = alignments[0].numpy().tolist()
            
            # Extract intervals using state machine
            intervals = []
            target_idx = 0
            start_frame = None
            end_frame = None
            saw_blank = False
            
            for t in range(T):
                token = path[t]
                if token == blank_id:
                    saw_blank = True
                    continue
                    
                if (target_idx + 1 < L and token == targets_list[target_idx + 1] and 
                    start_frame is not None and 
                    (targets_list[target_idx + 1] != targets_list[target_idx] or saw_blank)):
                    
                    intervals.append((start_frame, end_frame))
                    target_idx += 1
                    start_frame = t
                    end_frame = t
                    saw_blank = False
                elif target_idx < L and token == targets_list[target_idx]:
                    if start_frame is None:
                        start_frame = t
                    end_frame = t
                    saw_blank = False
                    
            if start_frame is not None:
                intervals.append((start_frame, end_frame))
        except Exception as e:
            print(f"Warning: torchaudio forced_align failed: {e}. Falling back to linear alignment.", flush=True)
            intervals = []
            step = T / max(L, 1)
            for idx in range(L):
                s = int(idx * step)
                e = int((idx + 1) * step) - 1
                intervals.append((s, max(s, e)))
            return intervals
            
        # Fallback padding
        while len(intervals) < L:
            if intervals:
                intervals.append(intervals[-1])
            else:
                intervals.append((0, T - 1))
                
        return intervals[:L]

    def compute_gop(self, 
                    log_probs: torch.Tensor, 
                    targets: torch.Tensor, 
                    intervals: List[Tuple[int, int]], 
                    vocab_tokens: List[str]) -> List[Dict]:
        """
        Computes Goodness of Pronunciation (GoP) log-likelihood ratios.
        """
        probs = torch.softmax(log_probs[0], dim=-1)
        
        L = targets.shape[1]
        targets_list = targets[0].cpu().numpy().tolist()
        
        results = []
        frame_stride_ms = 20.0
        
        for idx in range(L):
            token_id = targets_list[idx]
            phoneme = vocab_tokens[idx] if idx < len(vocab_tokens) else str(token_id)
            
            s_frame, e_frame = intervals[idx]
            token_probs = probs[s_frame:e_frame+1, token_id]
            
            if len(token_probs) > 0:
                log_probs_slice = torch.log(token_probs + 1e-8)
                gop_score = log_probs_slice.mean().item()
            else:
                gop_score = -10.0
                
            gop_prob = float(np.exp(gop_score))
            is_correct = bool(gop_prob >= 0.40)
            
            results.append({
                "phoneme": phoneme,
                "start_ms": float(s_frame * frame_stride_ms),
                "end_ms": float((e_frame + 1) * frame_stride_ms),
                "gop_prob": gop_prob,
                "is_correct": is_correct
            })
            
        return results