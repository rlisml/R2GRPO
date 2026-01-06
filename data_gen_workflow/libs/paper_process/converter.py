import os
import json
import shutil
import subprocess
import re
import logging
from pathlib import Path

# Configure log format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentConverter:
    def __init__(self, config: dict):
        self.config = config
        
        # --- 1. Robust Configuration Loading ---
        # Attempt to read from root or 'mineru_config' subsection
        if 'mineru_config' in config:
            self.model_dir = config['mineru_config'].get('model_dir')
            self.output_dir = Path(config['mineru_config'].get('tmp_dir', 'outputs/mineru_tmp'))
            self.backend = config['mineru_config'].get('backend', 'vlm-transformers')
        else:
            self.model_dir = config.get('model_dir')
            self.output_dir = Path(config.get('tmp_dir', 'outputs/mineru_tmp'))
            self.backend = config.get('backend', 'vlm-transformers')

        # Debug Log: Confirm received configuration
        if not self.model_dir:
            logging.error(f"!!! MinerU model_dir is MISSING. Received config keys: {list(config.keys())}")
        else:
            logging.info(f"MinerU initialized. Model Dir: {self.model_dir}")
            logging.info(f"Backend: {self.backend}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary input directory (stores renamed/sanitized files)
        self.temp_input_dir = self.output_dir / "_temp_inputs"
        self.temp_input_dir.mkdir(parents=True, exist_ok=True)

        if self.model_dir:
            self._setup_mineru_env()

    def _setup_mineru_env(self):
        """Configure magic-pdf.json"""
        home_dir = Path.home()
        config_path = home_dir / 'magic-pdf.json'
        
        target_config = {
            "models-dir": str(self.model_dir),
            "layout-config": {
                "model": "doclayout_yolo"
            },
            "formula-config": {
                "mfd_model": "yolo_v8_mfd",
                "mfr_model": "unimernet_small",
                "enable": True
            },
            "table-config": {
                "model": "rapid_table",
                "enable": False, 
                "max_time": 400
            }
        }

        # Read and merge configuration
        current_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    current_config = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read magic-pdf.json: {e}")

        # Write only if configuration differs
        if current_config.get('models-dir') != str(self.model_dir):
            logging.info(f"Updating MinerU config at {config_path}")
            current_config.update(target_config)
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to write magic-pdf.json: {e}")

    def _prepare_safe_input(self, original_path: Path) -> Path:
        """
        Sanitize filename and copy file to temporary directory.
        Fixes shell command failures caused by spaces or special characters (e.g., En-dash) in filenames.
        """
        # 1. Extract file suffix
        suffix = original_path.suffix
        
        # 2. Sanitize filename: Keep only alphanumeric characters and underscores
        # Replace unsafe characters with underscores
        stem = original_path.stem
        safe_stem = re.sub(r'[^\w\-]', '_', stem)
        
        # Prevent empty filenames
        if not safe_stem:
            safe_stem = "doc_file"
            
        safe_name = f"{safe_stem}{suffix}"
        safe_path = self.temp_input_dir / safe_name
        
        # 3. Copy file (skip if exists and size matches)
        if not safe_path.exists() or safe_path.stat().st_size != original_path.stat().st_size:
            shutil.copy2(original_path, safe_path)
            
        return safe_path

    def convert(self, file_path: str) -> str:
        original_path = Path(file_path).resolve()
        
        if not original_path.exists():
            logging.error(f"Input file not found: {original_path}")
            return None

        # --- Step 1: Prepare safe filename ---
        try:
            input_path = self._prepare_safe_input(original_path)
        except Exception as e:
            logging.error(f"Failed to prepare safe input file: {e}")
            return None

        # Expected output path
        file_stem = input_path.stem
        expected_output_dir = self.output_dir / file_stem
        expected_md_file = expected_output_dir / f"{file_stem}.md"

        if expected_md_file.exists():
            logging.info(f"Skipping existing file: {expected_md_file}")
            return str(expected_md_file)

        # --- Step 2: Build command ---
        cmd = [
            "mineru", 
            "-p", str(input_path),
            "-o", str(self.output_dir),
            "-m", "auto"
        ]
        
        # If using CUDA acceleration (usually required for vlm-transformers)
        if self.backend == "vlm-transformers":
            cmd.extend(["--device", "cuda"])

        logging.info(f"Running MinerU for: {original_path.name} (Safe name: {input_path.name})")
        env = os.environ.copy()
        
        # --- Step 3: Prepare Environment ---
        # Explicitly copy current environment variables and force overwrite MINERU_MODEL_SOURCE
        
        # Dynamically read source config, default to local
        source_mode = self.config.get('mineru_config', {}).get('source', 'local')
        env['MINERU_MODEL_SOURCE'] = source_mode

        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                env=env  # Explicitly pass environment
            )
            
            if expected_md_file.exists():
                logging.info(f"Conversion success: {expected_md_file}")
                return str(expected_md_file)
            else:
                logging.error(f"MinerU finished but output missing: {expected_md_file}")
                # Print detailed errors
                if result.stdout:
                    logging.error(f"STDOUT Tail: {result.stdout[-1000:]}")
                return None
                
        except subprocess.CalledProcessError as e:
            logging.error(f"MinerU conversion failed (Code {e.returncode})")
            logging.error(f"STDERR: {e.stderr}")
            return None