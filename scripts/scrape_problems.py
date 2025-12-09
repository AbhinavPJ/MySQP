'''
The purpose of this script is to download Hock–Schittkowski (HS) problems in APMonitor-style format
from the APMonitor website and save them locally for further processing.
'''
import requests  # type:ignore
import os
import time
def download(start_idx=1, end_idx=119, output_dir="../hs_problems"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_url = "https://apmonitor.com/wiki/uploads/Apps/"
    success_count = 0
    fail_count = 0
    for i in range(start_idx, end_idx + 1):
        problem_id = f"hs{i:03d}"
        filename = f"{problem_id}.apm"
        url = f"{base_url}{filename}"
        save_path = os.path.join(output_dir, filename)
        if os.path.exists(save_path):
            success_count += 1
            continue
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1
        time.sleep(1.0)
if __name__ == "__main__":
    download()