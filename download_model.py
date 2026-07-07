import os
import sys
import urllib.request
import time

def main():
    model_url = "https://huggingface.co/llmware/qwen3-4b-instruct-gguf/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    dest_dir = os.path.join("backend", "models")
    dest_path = os.path.join(dest_dir, "Qwen3-4B-Q4_K_M.gguf")

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    print(f"Starting download of: {model_url}")
    print(f"Saving to: {dest_path}")

    start_time = time.time()
    
    # Progress reporter callback
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        elapsed = time.time() - start_time
        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            total_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(
                f"\rProgress: {percent:.2f}% | {downloaded_mb:.1f}/{total_mb:.1f} MB | "
                f"Speed: {speed:.2f} MB/s | Elapsed: {elapsed:.1f}s"
            )
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(
                f"\rProgress: {downloaded_mb:.1f} MB downloaded | "
                f"Speed: {speed:.2f} MB/s | Elapsed: {elapsed:.1f}s"
            )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(model_url, dest_path, progress_callback)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError occurred during download: {e}")

if __name__ == "__main__":
    main()
