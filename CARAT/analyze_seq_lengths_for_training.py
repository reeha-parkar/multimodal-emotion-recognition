import torch
import numpy as np

def analyze_dataset_lengths(data):
    splits = ['train', 'val', 'test']
    for split in splits:
        if split not in data:
            continue
            
        split_data = data[split]
        texts = split_data['src-text']
        visuals = split_data['src-visual']
        audios = split_data['src-audio']
        
        # Get sequence lengths
        text_lengths = [text.shape[0] for text in texts]
        visual_lengths = [visual.shape[0] for visual in visuals]
        audio_lengths = [audio.shape[0] for audio in audios]
        
        print(f"\n{split.upper()} Split:")
        print(f"  Text lengths - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Mean: {np.mean(text_lengths):.2f}")
        print(f"  Visual lengths - Min: {min(visual_lengths)}, Max: {max(visual_lengths)}, Mean: {np.mean(visual_lengths):.2f}")
        print(f"  Audio lengths - Min: {min(audio_lengths)}, Max: {max(audio_lengths)}, Mean: {np.mean(audio_lengths):.2f}")
        
    # Recommend CTC target length
    all_text_lengths = []
    for split in splits:
        if split in data:
            texts = data[split]['src-text']
            all_text_lengths.extend([text.shape[0] for text in texts])
    
    max_text_len = max(all_text_lengths)
    percentile_95 = np.percentile(all_text_lengths, 95)
    
    print(f"\nRECOMMENDATIONS:")
    print(f"  Maximum text length in dataset: {max_text_len}")
    print(f"  95th percentile text length: {percentile_95:.0f}")
    print(f"  Recommended CTC target length: {max(int(percentile_95), 50)}")


def get_optimal_configurations(data):
    all_text_lengths = []
    all_visual_lengths = []
    all_audio_lengths = []
    
    for split in ['train', 'val', 'test']:
        if split in data:
            all_text_lengths.extend([seq.shape[0] for seq in data[split]['src-text']])
            all_visual_lengths.extend([seq.shape[0] for seq in data[split]['src-visual']])
            all_audio_lengths.extend([seq.shape[0] for seq in data[split]['src-audio']])
    
    # Statistical analysis
    text_stats = {
        'max': max(all_text_lengths),
        'p95': int(np.percentile(all_text_lengths, 95)),
        'p99': int(np.percentile(all_text_lengths, 99)),
        'mean': np.mean(all_text_lengths)
    }
    
    visual_stats = {
        'max': max(all_visual_lengths),
        'p95': int(np.percentile(all_visual_lengths, 95)),
        'p99': int(np.percentile(all_visual_lengths, 99)),
        'mean': np.mean(all_visual_lengths)
    }
    
    audio_stats = {
        'max': max(all_audio_lengths),
        'p95': int(np.percentile(all_audio_lengths, 95)),
        'p99': int(np.percentile(all_audio_lengths, 99)),
        'mean': np.mean(all_audio_lengths)
    }
    
    print(f"Text    - Max: {text_stats['max']}, 95%: {text_stats['p95']}, 99%: {text_stats['p99']}, Mean: {text_stats['mean']:.1f}")
    print(f"Visual  - Max: {visual_stats['max']}, 95%: {visual_stats['p95']}, 99%: {visual_stats['p99']}, Mean: {visual_stats['mean']:.1f}")
    print(f"Audio   - Max: {audio_stats['max']}, 95%: {audio_stats['p95']}, 99%: {audio_stats['p99']}, Mean: {audio_stats['mean']:.1f}")
    
    # Production recommendations (based on 99th percentile + margin)
    ctc_target = max(visual_stats['p99'], audio_stats['p99']) + 20
    text_max_pos = text_stats['p99'] + 20
    visual_max_pos = ctc_target + 30  # CTC target + safety margin
    audio_max_pos = ctc_target + 30   # CTC target + safety margin
  
    print(f"\n------------- Recommendations (99 %ile CTC + margin) -------------:")
    print(f"   --ctc_target_length: {ctc_target}")
    print(f"   --text_max_position_embeddings: {text_max_pos}")
    print(f"   --visual_max_position_embeddings: {visual_max_pos}")
    print(f"   --audio_max_position_embeddings: {audio_max_pos}")

    # Conservative values
    ctc_target_safe = max(visual_stats['max'], audio_stats['max']) + 20
    text_max_pos_safe = text_stats['max'] + 20
    visual_max_pos_safe = ctc_target_safe + 30
    audio_max_pos_safe = ctc_target_safe + 30

    print(f"\n------------- Safe Values for Production (based on max) -------------:")
    print(f"   --ctc_target_length: {ctc_target_safe}")
    print(f"   --text_max_position_embeddings: {text_max_pos_safe}")
    print(f"   --visual_max_position_embeddings: {visual_max_pos_safe}")
    print(f"   --audio_max_position_embeddings: {audio_max_pos_safe}")


if __name__ == "__main__":
    try:
        # Load the dataset
        data = torch.load('./data/omg_multitask.pt')

        analyze_dataset_lengths(data)
        print("\n------------------------------------------------------")
        get_optimal_configurations(data)

    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()

    
