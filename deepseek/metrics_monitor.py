import time
import psutil
import csv
import os
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available. GPU monitoring disabled.")

class MetricsMonitor:
    def __init__(self, log_file: Optional[str] = None):

        if log_file and os.path.isabs(log_file):
            self.log_file = log_file
            self.logs_dir = os.path.dirname(log_file)
        else:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(module_dir)
            default_logs_dir = os.environ.get("VOICE_BRIDGE_LOG_DIR") or os.path.join(repo_root, "logs")
            os.makedirs(default_logs_dir, exist_ok=True)
            self.logs_dir = default_logs_dir
            
            if log_file:
                self.log_file = os.path.join(self.logs_dir, log_file)
            else:
                self.log_file = os.path.join(self.logs_dir, "metrics_log.csv")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.current_session = {
            'wake_word_detected_time': None,
            'recording_start_time': None,
            'recording_end_time': None,
            'transcription_start_time': None,
            'transcription_end_time': None,
            'intent_classification_start_time': None,
            'intent_classification_end_time': None,
            'execution_start_time': None,
            'execution_end_time': None,
            'total_start_time': None,
            'total_end_time': None
        }
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'wake_word',
                    'transcribed_text',
                    'command',
                    'intent',
                    'query',
                    'recording_latency',
                    'transcription_latency',
                    'intent_classification_latency',
                    'execution_latency',
                    'total_latency',
                    'wake_to_execution_latency',
                    'cpu_percent',
                    'memory_mb',
                    'memory_percent',
                    'gpu_utilization',
                    'gpu_memory_mb',
                    'audio_duration',
                    'audio_size_kb',
                    'wer',
                    'notes'
                ])
        
        print(f"Metrics logging to: {self.log_file}")
        
    def mark_wake_word_detected(self):
        self.current_session['wake_word_detected_time'] = time.time()
        self.current_session['total_start_time'] = time.time()
    
    def start_recording(self):
        self.current_session['recording_start_time'] = time.time()
    
    def end_recording(self):
        self.current_session['recording_end_time'] = time.time()
    
    def start_transcription(self):
        self.current_session['transcription_start_time'] = time.time()
    
    def end_transcription(self):
        self.current_session['transcription_end_time'] = time.time()
    
    def start_intent_classification(self):
        self.current_session['intent_classification_start_time'] = time.time()
    
    def end_intent_classification(self):
        self.current_session['intent_classification_end_time'] = time.time()
    
    def start_execution(self):
        self.current_session['execution_start_time'] = time.time()
    
    def end_execution(self):
        self.current_session['execution_end_time'] = time.time()
        self.current_session['total_end_time'] = time.time()
    
    def get_recording_latency(self) -> Optional[float]:
        if self.current_session['recording_start_time'] and self.current_session['recording_end_time']:
            return self.current_session['recording_end_time'] - self.current_session['recording_start_time']
        return None
    
    def get_transcription_latency(self) -> Optional[float]:
        if self.current_session['transcription_start_time'] and self.current_session['transcription_end_time']:
            return self.current_session['transcription_end_time'] - self.current_session['transcription_start_time']
        return None
    
    def get_intent_classification_latency(self) -> Optional[float]:
        if self.current_session['intent_classification_start_time'] and self.current_session['intent_classification_end_time']:
            return self.current_session['intent_classification_end_time'] - self.current_session['intent_classification_start_time']
        return None
    
    def get_execution_latency(self) -> Optional[float]:
        if self.current_session['execution_start_time'] and self.current_session['execution_end_time']:
            return self.current_session['execution_end_time'] - self.current_session['execution_start_time']
        return None
    
    def get_total_latency(self) -> Optional[float]:
        if self.current_session['total_start_time'] and self.current_session['total_end_time']:
            return self.current_session['total_end_time'] - self.current_session['total_start_time']
        return None
    
    def get_wake_to_execution_latency(self) -> Optional[float]:
        if self.current_session['wake_word_detected_time'] and self.current_session['execution_end_time']:
            return self.current_session['execution_end_time'] - self.current_session['wake_word_detected_time']
        return None
    
    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info()
        
        return {
            'total_mb': mem.total / (1024 * 1024),
            'available_mb': mem.available / (1024 * 1024),
            'used_mb': mem.used / (1024 * 1024),
            'percent': mem.percent,
            'process_mb': process_mem.rss / (1024 * 1024)
        }
    
    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu = gpus[0]
            return {
                'utilization': gpu.load * 100,
                'memory_used_mb': gpu.memoryUsed,
                'memory_total_mb': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                'temperature': gpu.temperature
            }
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return None
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        
        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
        return round(wer, 4)
    
    def log_session(self, 
                   wake_word: str,
                   transcribed_text: str,
                   command: str,
                   intent: str,
                   query: str,
                   audio_data: Optional[np.ndarray] = None,
                   ground_truth: Optional[str] = None,
                   notes: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        recording_latency = self.get_recording_latency()
        transcription_latency = self.get_transcription_latency()
        intent_classification_latency = self.get_intent_classification_latency()
        execution_latency = self.get_execution_latency()
        total_latency = self.get_total_latency()
        wake_to_exec_latency = self.get_wake_to_execution_latency()
        
        cpu_percent = self.get_cpu_usage()
        memory = self.get_memory_usage()
        gpu = self.get_gpu_usage()
        
        audio_duration = recording_latency if recording_latency else 0
        audio_size_kb = (audio_data.nbytes / 1024) if audio_data is not None else 0
        
        wer = self.calculate_wer(ground_truth, transcribed_text) if ground_truth else None
        
        fmt_latency = lambda v: f"{v:.6f}" if v is not None else ''

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                wake_word,
                transcribed_text,
                command,
                intent,
                query,
                fmt_latency(recording_latency),
                fmt_latency(transcription_latency),
                fmt_latency(intent_classification_latency),
                fmt_latency(execution_latency),
                fmt_latency(total_latency),
                fmt_latency(wake_to_exec_latency),
                round(cpu_percent, 2),
                round(memory['process_mb'], 2),
                round(memory['percent'], 2),
                round(gpu['utilization'], 2) if gpu else '',
                round(gpu['memory_used_mb'], 2) if gpu else '',
                fmt_latency(audio_duration),
                round(audio_size_kb, 2),
                wer if wer is not None else '',
                notes
            ])
        
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Transcribed: {transcribed_text}")
        print(f"Intent: {intent} | Query: {query}")
        print(f"\nLATENCY:")
        print(f"   Recording:      {recording_latency:.3f}s" if recording_latency else "   Recording:      N/A")
        print(f"   Transcription:  {transcription_latency:.3f}s" if transcription_latency else "   Transcription:  N/A")
        print(f"   Intent Class.:  {intent_classification_latency:.3f}s" if intent_classification_latency else "   Intent Class.:  N/A")
        print(f"   Execution:      {execution_latency:.3f}s" if execution_latency else "   Execution:      N/A")
        print(f"   Total:          {total_latency:.3f}s" if total_latency else "   Total:          N/A")
        print(f"   Wake-Execute:   {wake_to_exec_latency:.3f}s" if wake_to_exec_latency else "   Wake-Execute:   N/A")
        print(f"\nPERFORMANCE:")
        print(f"   CPU:            {cpu_percent:.1f}%")
        print(f"   Memory:         {memory['process_mb']:.1f} MB ({memory['percent']:.1f}%)")
        if gpu:
            print(f"   GPU:            {gpu['utilization']:.1f}%")
            print(f"   GPU Memory:     {gpu['memory_used_mb']:.1f} MB")
        print(f"\nAUDIO:")
        print(f"   Duration:       {audio_duration:.3f}s")
        print(f"   Size:           {audio_size_kb:.2f} KB")
        if wer is not None:
            print(f"\nACCURACY:")
            print(f"   WER:            {wer:.4f} ({wer*100:.2f}%)")
            print(f"   Ground Truth:   {ground_truth}")
        print("="*60 + "\n")
        
        self.reset_session()
    
    def reset_session(self):
        self.current_session = {
            'wake_word_detected_time': None,
            'recording_start_time': None,
            'recording_end_time': None,
            'transcription_start_time': None,
            'transcription_end_time': None,
            'intent_classification_start_time': None,
            'intent_classification_end_time': None,
            'execution_start_time': None,
            'execution_end_time': None,
            'total_start_time': None,
            'total_end_time': None
        }
    
    def generate_summary_report(self, output_file: str = "metrics_summary.txt"):
        if not os.path.exists(self.log_file):
            print("No metrics log file found")
            return
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("No data in metrics log")
            return
        
        total_sessions = len(data)
        
        recording_latencies = [float(row['recording_latency']) for row in data if row['recording_latency']]
        transcription_latencies = [float(row['transcription_latency']) for row in data if row['transcription_latency']]
        total_latencies = [float(row['total_latency']) for row in data if row['total_latency']]
        
        cpu_usages = [float(row['cpu_percent']) for row in data if row['cpu_percent']]
        memory_usages = [float(row['memory_mb']) for row in data if row['memory_mb']]
        
        wers = [float(row['wer']) for row in data if row['wer']]
        
        target_output = os.path.join(self.logs_dir, output_file) if not os.path.isabs(output_file) else output_file
        with open(target_output, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VOICE ASSISTANT METRICS SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Sessions: {total_sessions}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("LATENCY STATISTICS (seconds)\n")
            f.write("-" * 70 + "\n")
            if recording_latencies:
                f.write(f"Recording:\n")
                f.write(f"  Average: {np.mean(recording_latencies):.4f}s\n")
                f.write(f"  Median:  {np.median(recording_latencies):.4f}s\n")
                f.write(f"  Min:     {np.min(recording_latencies):.4f}s\n")
                f.write(f"  Max:     {np.max(recording_latencies):.4f}s\n\n")
            
            if transcription_latencies:
                f.write(f"Transcription:\n")
                f.write(f"  Average: {np.mean(transcription_latencies):.4f}s\n")
                f.write(f"  Median:  {np.median(transcription_latencies):.4f}s\n")
                f.write(f"  Min:     {np.min(transcription_latencies):.4f}s\n")
                f.write(f"  Max:     {np.max(transcription_latencies):.4f}s\n\n")
            
            if total_latencies:
                f.write(f"Total (End-to-End):\n")
                f.write(f"  Average: {np.mean(total_latencies):.4f}s\n")
                f.write(f"  Median:  {np.median(total_latencies):.4f}s\n")
                f.write(f"  Min:     {np.min(total_latencies):.4f}s\n")
                f.write(f"  Max:     {np.max(total_latencies):.4f}s\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 70 + "\n")
            if cpu_usages:
                f.write(f"CPU Usage:\n")
                f.write(f"  Average: {np.mean(cpu_usages):.2f}%\n")
                f.write(f"  Median:  {np.median(cpu_usages):.2f}%\n")
                f.write(f"  Min:     {np.min(cpu_usages):.2f}%\n")
                f.write(f"  Max:     {np.max(cpu_usages):.2f}%\n\n")
            
            if memory_usages:
                f.write(f"Memory Usage:\n")
                f.write(f"  Average: {np.mean(memory_usages):.2f} MB\n")
                f.write(f"  Median:  {np.median(memory_usages):.2f} MB\n")
                f.write(f"  Min:     {np.min(memory_usages):.2f} MB\n")
                f.write(f"  Max:     {np.max(memory_usages):.2f} MB\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("ACCURACY STATISTICS\n")
            f.write("-" * 70 + "\n")
            if wers:
                f.write(f"Word Error Rate (WER):\n")
                f.write(f"  Average: {np.mean(wers):.4f} ({np.mean(wers)*100:.2f}%)\n")
                f.write(f"  Median:  {np.median(wers):.4f} ({np.median(wers)*100:.2f}%)\n")
                f.write(f"  Min:     {np.min(wers):.4f} ({np.min(wers)*100:.2f}%)\n")
                f.write(f"  Max:     {np.max(wers):.4f} ({np.max(wers)*100:.2f}%)\n")
                f.write(f"  Sessions with WER data: {len(wers)}/{total_sessions}\n\n")
            else:
                f.write("No WER data available (ground truth not provided)\n\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"Summary report generated: {target_output}")


def create_monitor(log_file: str = "metrics_log.csv") -> MetricsMonitor:
    return MetricsMonitor(log_file)
