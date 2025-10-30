# Download RobotWin Assets

Run this command to download and extract all required assets:

```bash
cd ~/RoboTwin
source ~/.config/hrdt/activate.sh
bash script/_download_assets.sh
```

This will:
1. Download 3 zip files from HuggingFace (~several GB):
   - `background_texture.zip`
   - `embodiments.zip`
   - `objects.zip`
2. Extract all zip files
3. Remove the zip files
4. Configure asset paths

**Note:** This download may take 20-30 minutes depending on your internet speed.

## Manual Download (Alternative)

If you prefer to download manually:

```bash
cd ~/RoboTwin/assets
source ~/.config/hrdt/activate.sh
python _download.py

# Extract files
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip

# Clean up
rm *.zip

# Configure paths
cd ..
python ./script/update_embodiment_config_path.py
```

## Verification

After download completes, verify assets are present:

```bash
ls -lh ~/RoboTwin/assets/objects/objaverse/list.json  # Should exist
ls -d ~/RoboTwin/assets/embodiments/*/               # Should list embodiment folders
```

