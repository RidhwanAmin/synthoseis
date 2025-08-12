# Synthoseis Makefile
# Provides convenient commands to run seismic data generation with different configurations

# Default configuration
OUTPUT_DIR := ./output
VERBOSE := --verbose
PYTHON := python

# Configuration file paths
SMOOTH_CONFIG := ./config/smooth/example.json
FLAT_CLEAN_CONFIG := ./config/flat_clean_config_updated.json
STRUCTURED_CONFIG := ./config/structured_realistic_config_updated.json
FLAT_CLEAN_RANGE := ./config/flat_clean_config_range.json
STRUCTURED_RANGE := ./config/structured_realistic_config_range.json

# Default target
.PHONY: help
help:
	@echo "Synthoseis Makefile - Seismic Data Generation"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  run-smooth        - Run with smooth configuration"
	@echo "  run-flat-clean    - Run with flat clean configuration"
	@echo "  run-structured    - Run with structured realistic configuration"
	@echo "  run-flat-range    - Run with flat clean range configuration"
	@echo "  run-struct-range  - Run with structured range configuration"
	@echo "  run-custom        - Run with custom config (requires CONFIG variable)"
	@echo ""
	@echo "With specific run ID:"
	@echo "  run-smooth-id     - Run smooth with specific ID (requires ID variable)"
	@echo "  run-flat-id       - Run flat clean with specific ID (requires ID variable)"
	@echo "  run-struct-id     - Run structured with specific ID (requires ID variable)"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean             - Clean output directory"
	@echo "  show-configs      - Show available configuration files"
	@echo "  list-outputs      - List generated output files"
	@echo ""
	@echo "Usage examples:"
	@echo "  make run-smooth                    # Run with smooth config"
	@echo "  make run-smooth-id ID=123         # Run with smooth config and ID 123"
	@echo "  make run-custom CONFIG=./my.json  # Run with custom config file"
	@echo "  make OUTPUT_DIR=./my_output run-smooth  # Custom output directory"

# Main run targets with verbose output and custom output directory
.PHONY: run-smooth
run-smooth:
	@echo "Running Synthoseis with smooth configuration..."
	@echo "Config: $(SMOOTH_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(SMOOTH_CONFIG) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-flat-clean
run-flat-clean:
	@echo "Running Synthoseis with flat clean configuration..."
	@echo "Config: $(FLAT_CLEAN_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(FLAT_CLEAN_CONFIG) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-structured
run-structured:
	@echo "Running Synthoseis with structured realistic configuration..."
	@echo "Config: $(STRUCTURED_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(STRUCTURED_CONFIG) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-flat-range
run-flat-range:
	@echo "Running Synthoseis with flat clean range configuration..."
	@echo "Config: $(FLAT_CLEAN_RANGE)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(FLAT_CLEAN_RANGE) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-struct-range
run-struct-range:
	@echo "Running Synthoseis with structured range configuration..."
	@echo "Config: $(STRUCTURED_RANGE)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(STRUCTURED_RANGE) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-custom
run-custom:
ifndef CONFIG
	@echo "Error: CONFIG variable is required"
	@echo "Usage: make run-custom CONFIG=./path/to/config.json"
	@exit 1
endif
	@echo "Running Synthoseis with custom configuration..."
	@echo "Config: $(CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(CONFIG) $(VERBOSE) --output-dir $(OUTPUT_DIR)

# Run targets with specific ID
.PHONY: run-smooth-id
run-smooth-id:
ifndef ID
	@echo "Error: ID variable is required"
	@echo "Usage: make run-smooth-id ID=123"
	@exit 1
endif
	@echo "Running Synthoseis with smooth configuration and ID $(ID)..."
	@echo "Config: $(SMOOTH_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(SMOOTH_CONFIG) --run-id $(ID) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-flat-id
run-flat-id:
ifndef ID
	@echo "Error: ID variable is required"
	@echo "Usage: make run-flat-id ID=123"
	@exit 1
endif
	@echo "Running Synthoseis with flat clean configuration and ID $(ID)..."
	@echo "Config: $(FLAT_CLEAN_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(FLAT_CLEAN_CONFIG) --run-id $(ID) $(VERBOSE) --output-dir $(OUTPUT_DIR)

.PHONY: run-struct-id
run-struct-id:
ifndef ID
	@echo "Error: ID variable is required"
	@echo "Usage: make run-struct-id ID=123"
	@exit 1
endif
	@echo "Running Synthoseis with structured configuration and ID $(ID)..."
	@echo "Config: $(STRUCTURED_CONFIG)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON) run_synthosei2.py --config $(STRUCTURED_CONFIG) --run-id $(ID) $(VERBOSE) --output-dir $(OUTPUT_DIR)

# Quick run target (uses default /output directory)
.PHONY: quick-run
quick-run:
	@echo "Quick run with smooth configuration to /output..."
	$(PYTHON) run_synthosei2.py --config $(SMOOTH_CONFIG) $(VERBOSE) --output-dir /output

# Utility targets
.PHONY: clean
clean:
	@echo "Cleaning output directory: $(OUTPUT_DIR)"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		rm -rf $(OUTPUT_DIR)/*; \
		echo "Output directory cleaned"; \
	else \
		echo "Output directory does not exist"; \
	fi

.PHONY: show-configs
show-configs:
	@echo "Available configuration files:"
	@echo "=============================="
	@echo "Smooth configs:"
	@find ./config -name "*.json" -path "*/smooth/*" | sort
	@echo ""
	@echo "Main configs:"
	@find ./config -maxdepth 1 -name "*.json" | sort
	@echo ""
	@echo "Other configs:"
	@find ./config -name "*.json" -path "*/other/*" | head -5
	@if [ $$(find ./config -name "*.json" -path "*/other/*" | wc -l) -gt 5 ]; then \
		echo "... and more in ./config/other/"; \
	fi

.PHONY: list-outputs
list-outputs:
	@echo "Generated output files:"
	@echo "======================"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		find $(OUTPUT_DIR) -type f \( -name "*.npy" -o -name "*.png" \) | sort; \
	else \
		echo "No output directory found at $(OUTPUT_DIR)"; \
	fi

.PHONY: create-output
create-output:
	@echo "Creating output directory structure..."
	@mkdir -p $(OUTPUT_DIR)/smooth
	@mkdir -p $(OUTPUT_DIR)/faulty
	@echo "Created: $(OUTPUT_DIR)/smooth"
	@echo "Created: $(OUTPUT_DIR)/faulty"

# Environment setup targets
.PHONY: check-env
check-env:
	@echo "Checking environment..."
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Required files:"
	@if [ -f "run_synthosei2.py" ]; then echo "✓ run_synthosei2.py found"; else echo "✗ run_synthosei2.py missing"; fi
	@if [ -f "main.py" ]; then echo "✓ main.py found"; else echo "✗ main.py missing"; fi
	@if [ -d "config" ]; then echo "✓ config directory found"; else echo "✗ config directory missing"; fi
	@if [ -d "datagenerator" ]; then echo "✓ datagenerator directory found"; else echo "✗ datagenerator directory missing"; fi

# Test targets
.PHONY: test-smooth
test-smooth:
	@echo "Testing smooth configuration (dry run)..."
	$(PYTHON) run_synthosei2.py --config $(SMOOTH_CONFIG) --help

# Advanced usage examples as targets
.PHONY: example-batch
example-batch:
	@echo "Example: Running batch of configurations..."
	@echo "This would run multiple configs in sequence:"
	@echo "make run-smooth"
	@echo "make run-flat-clean" 
	@echo "make run-structured"

# Override default variables
# Usage: make OUTPUT_DIR=/custom/path run-smooth
# Usage: make VERBOSE= run-smooth  (to disable verbose)
# Usage: make PYTHON=python3 run-smooth
