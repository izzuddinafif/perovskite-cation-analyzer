import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================
# Functions
# =============================================
def extract_sample_info(filename):
    """Extract sample name, concentration, and unit from filename."""
    # Pattern to extract: sample name, concentration, unit
    pattern = r"(.+)_(\d+\.\d+)_(\w+)"
    match = re.match(pattern, os.path.splitext(filename)[0])
    if match:
        sample_name = match.group(1)
        concentration = float(match.group(2))
        unit = match.group(3)
        return sample_name, concentration, unit
    else:
        raise ValueError(f"Could not parse sample info from filename: {filename}")

def parse_simple_csv(filepath):
    """Parse simple CSV format with X/Y or Wavelength/Absorbance columns."""
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            # First try to read directly as a CSV
            df = pd.read_csv(filepath, encoding=encoding, sep=None, engine='python')
            
            # Debug print the first few rows and dtypes
            print(f"DataFrame head:\n{df.head()}")
            print(f"DataFrame dtypes:\n{df.dtypes}")
            
            # Check column names
            columns = df.columns.tolist()
            
            # Try to identify wavelength and absorbance columns
            wavelength_col = None
            absorbance_col = None
            
            # Look for common names
            for col in columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['wavelength', 'lambda', 'nm', 'x']):
                    wavelength_col = col
                elif any(x in col_lower for x in ['absorbance', 'abs', 'a', 'y']):
                    absorbance_col = col
            
            # If we found the columns, use them
            if wavelength_col and absorbance_col:
                df = df[[wavelength_col, absorbance_col]].copy()
                df.columns = ['Wavelength', 'Absorbance']
            # If we couldn't identify columns but have exactly 2, assume first is wavelength and second is absorbance
            elif len(columns) == 2:
                df.columns = ['Wavelength', 'Absorbance']
            
            # Ensure the columns are numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values that might have resulted from conversion
            df = df.dropna()
            
            if not df.empty:
                return df
                
        except Exception as e:
            print(f"Exception in pandas CSV reading: {str(e)}")
            # Try the next encoding
            continue
    
    # If we get here, try a more manual approach
    try:
        wavelengths = []
        absorbances = []
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Skip potential header - start from line that has two numeric values
        for line in lines:
            # Remove any quotes
            line = line.replace('"', '').replace("'", "")
            
            # Split by common delimiters
            for delimiter in ['\t', ',', ';', ' ']:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    try:
                        # Try to convert both parts to float
                        wavelength = float(parts[0])
                        absorbance = float(parts[1])
                        wavelengths.append(wavelength)
                        absorbances.append(absorbance)
                        break  # Successfully parsed this line
                    except ValueError:
                        # Not numeric values, try next delimiter
                        continue
        
        if wavelengths:
            return pd.DataFrame({
                'Wavelength': wavelengths,
                'Absorbance': absorbances
            })
    except Exception as e:
        print(f"Exception in manual parsing: {str(e)}")
    
    # If we get here, none of the approaches worked
    raise ValueError(f"Could not read file format: {filepath}")

def parse_jasco_csv(filepath):
    """Parse JASCO CSV format with metadata headers with multiple encoding attempts."""
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                lines = file.readlines()
            
            # Find where the actual spectral data begins
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.strip() == "XYDATA":
                    data_start_idx = i + 1
                    break
            
            if data_start_idx is None:
                continue  # Try next encoding if XYDATA not found
            
            # Parse the spectral data
            wavelengths = []
            absorbances = []
            
            for line in lines[data_start_idx:]:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split()
                    if len(parts) >= 2:  # Ensure there are at least two values
                        try:
                            wavelength = float(parts[0])
                            absorbance = float(parts[1])
                            wavelengths.append(wavelength)
                            absorbances.append(absorbance)
                        except ValueError:
                            # Skip lines that don't have proper float values
                            continue
            
            if wavelengths:  # If we found data, return it
                return pd.DataFrame({
                    'Wavelength': wavelengths,
                    'Absorbance': absorbances
                })
                
        except Exception as e:
            # Try the next encoding
            continue
    
    # If we get here, none of the encodings worked
    raise ValueError(f"Could not read file with any supported encoding: {filepath}")

def get_absorbance_at_wavelength(df, target_wavelength):
    """Extract absorbance closest to a target wavelength."""
    if df.empty:
        raise ValueError("DataFrame is empty, no spectral data found")
    
    # Ensure all values are numeric
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    df['Absorbance'] = pd.to_numeric(df['Absorbance'], errors='coerce')
    df = df.dropna()
    
    if df.empty:
        raise ValueError("No valid numeric data found after conversion")
    
    # Debug print
    print(f"DataFrame dtypes after conversion:\n{df.dtypes}")
    print(f"Target wavelength: {target_wavelength}, type: {type(target_wavelength)}")
    print(f"First few wavelengths: {df['Wavelength'].head().tolist()}")
    
    # Find the closest wavelength
    idx = (df['Wavelength'] - target_wavelength).abs().idxmin()
    actual_wavelength = df.loc[idx, 'Wavelength']
    absorbance = df.loc[idx, 'Absorbance']
    
    return actual_wavelength, absorbance

def predict_concentration(absorbance_unknown, slope, intercept):
    """Convert an absorbance reading to concentration using the calibration curve."""
    return (absorbance_unknown - intercept) / slope

def calculate_molar_absorptivity(absorbance, concentration, path_length=1.0):
    """
    Calculate molar absorptivity (ε) using Beer's Law: ε = A / (c * l)
    
    Parameters:
    absorbance (float): Measured absorbance (A)
    concentration (float): Concentration of the solution (mol/L)
    path_length (float): Cuvette path length (cm), default is 1 cm.
    
    Returns:
    float: Molar absorptivity (L·mol⁻¹·cm⁻¹)
    """
    if concentration <= 0 or path_length <= 0:
        raise ValueError("Concentration and path length must be positive values.")
    
    molar_absorptivity = absorbance / (concentration * path_length)
    return molar_absorptivity

def convert_concentration_to_molar(concentration, unit):
    """
    Convert concentration to mol/L (M) if not already in that unit.
    Returns the converted concentration and a boolean indicating if conversion was done.
    """
    # Clean and normalize the unit string
    unit_cleaned = unit.lower().strip().replace('μ', 'u')  # Normalize micro symbol
    
    # Check if already in molar units
    if unit_cleaned in ['m', 'mol/l', 'mol/l', 'molar']:
        return concentration, True
    
    # Handle various unit formats
    if unit_cleaned in ['mm', 'mmol/l', 'mmol/l', 'millimolar']:
        return concentration * 1e-3, True
    elif unit_cleaned in ['um', 'µm', 'umol/l', 'µmol/l', 'micromolar', 'μm']:
        return concentration * 1e-6, True
    elif unit_cleaned in ['nm', 'nmol/l', 'nmol/l', 'nanomolar']:
        return concentration * 1e-9, True
    
    # Additional check for non-standard formats
    if 'u' in unit_cleaned or 'µ' in unit_cleaned or 'μ' in unit_cleaned:
        if 'm' in unit_cleaned:
            return concentration * 1e-6, True
    
    # If unit is not recognized as molar, return original and False
    return concentration, False

# =============================================
# Alternative parsing function for problematic files
# =============================================
def parse_csv_binary(filepath, target_wavelength):
    """Parse CSV in binary mode, looking for wavelength and absorbance values."""
    try:
        # Try to read file in binary mode
        with open(filepath, 'rb') as f:
            binary_data = f.read()
        
        # Convert binary to string, replacing problematic characters
        text_data = binary_data.decode('ascii', errors='replace')
        lines = text_data.split('\n')
        
        # Process each line
        wavelengths = []
        absorbances = []
        
        for line in lines:
            # Skip empty lines or likely header lines
            if not line.strip() or not re.search(r'^\s*[\d\.\-]+\s+[\d\.\-]+', line):
                continue
                
            # Extract data
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    wavelength = float(parts[0])
                    absorbance = float(parts[1])
                    wavelengths.append(wavelength)
                    absorbances.append(absorbance)
                except ValueError:
                    continue
        
        if not wavelengths:
            return None, None
            
        # Find closest wavelength to target
        df = pd.DataFrame({'Wavelength': wavelengths, 'Absorbance': absorbances})
        return get_absorbance_at_wavelength(df, target_wavelength)
        
    except Exception as e:
        print(f"Binary parsing error: {str(e)}")
        return None, None

# =============================================
# Main Program
# =============================================
def main():
    print("=" * 60)
    print("UV-Vis Calibration and Molar Absorptivity Calculator")
    print("=" * 60)
    
    # Get user input for data directory
    while True:
        data_dir = input("Enter the path to your data files: ").strip()
        if os.path.exists(data_dir):
            break
        print(f"Error: Directory '{data_dir}' does not exist. Please try again.")
    
    # Get target wavelength
    while True:
        try:
            target_wavelength = float(input("Enter the target wavelength (nm): ").strip())
            break
        except ValueError:
            print("Error: Please enter a valid number for wavelength.")
    
    # Get cuvette path length
    while True:
        try:
            path_length = float(input("Enter the cuvette path length (cm, default 1.0): ").strip() or "1.0")
            if path_length <= 0:
                print("Error: Path length must be positive.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid number for path length.")
    
    # Collect data from files
    sample_names = []
    concentrations = []
    units = []
    wavelengths = []
    absorbances = []
    filenames = []
    
    print("\nProcessing files...")
    
    # Process each file
    file_count = 0
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith((".csv", ".txt")):
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Extract sample information from filename
                sample_name, concentration, unit = extract_sample_info(filename)
                
                # Try all parsing methods in sequence until one works
                df = None
                parsing_methods = [
                    ("Simple CSV", parse_simple_csv),
                    ("JASCO format", parse_jasco_csv)
                ]
                
                for method_name, parser_func in parsing_methods:
                    try:
                        print(f"Trying {method_name} parser for {filename}...")
                        df = parser_func(filepath)
                        if df is not None and not df.empty:
                            print(f"Successfully parsed with {method_name} parser")
                            break
                    except Exception as e:
                        print(f"  {method_name} parsing failed: {str(e)}")
                
                # If all parsing methods failed, try binary method
                if df is None or df.empty:
                    print(f"Trying binary parser for {filename}...")
                    actual_wavelength, absorbance = parse_csv_binary(filepath, target_wavelength)
                    
                    if actual_wavelength is not None:
                        print(f"Successfully parsed with binary parser")
                    else:
                        raise ValueError("Failed to extract data with all parsing methods")
                else:
                    # Get absorbance at target wavelength
                    actual_wavelength, absorbance = get_absorbance_at_wavelength(df, target_wavelength)
                
                # Store the data
                sample_names.append(sample_name)
                concentrations.append(concentration)
                units.append(unit)
                wavelengths.append(actual_wavelength)
                absorbances.append(absorbance)
                filenames.append(filename)
                
                print(f"Processed {filename}:")
                print(f"  Sample: {sample_name}")
                print(f"  Concentration: {concentration} {unit}")
                print(f"  Absorbance at {actual_wavelength} nm: {absorbance:.4f}")
                file_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Check if we have data to process
    if file_count == 0:
        print("\nNo valid data files were found. Please check your file naming format and directory.")
        return
    
    # Create DataFrame
    data = pd.DataFrame({
        "Filename": filenames,
        "Sample": sample_names,
        "Concentration": concentrations,
        "Unit": units,
        "Wavelength": wavelengths,
        "Absorbance": absorbances
    })
    
    # Check if all units are the same
    if len(set(units)) > 1:
        print(f"\nWarning: Multiple concentration units found: {set(units)}")
        print("The calibration curve may not be accurate if units are inconsistent.")
        concentration_label = "Concentration (mixed units)"
    else:
        concentration_label = f"Concentration ({units[0]})"
    
    # Sort by concentration for better visualization
    data = data.sort_values("Concentration")
    
    # Display the collected data
    print("\nCollected Data:")
    print(data[["Sample", "Concentration", "Unit", "Absorbance"]].to_string(index=False))
    
    # Check if all samples are the same type
    unique_samples = set(sample_names)
    if len(unique_samples) > 1:
        print(f"\nWarning: Multiple sample types detected: {unique_samples}")
        print("Make sure all samples follow the same Beer-Lambert behavior.")
        sample_label = "Multiple Samples"
    else:
        sample_label = sample_names[0]
    
    # Fit calibration curve
    X = data["Concentration"].values.reshape(-1, 1)
    y = data["Absorbance"].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = r2_score(y, model.predict(X))
    
    print(f"\nCalibration Curve: A = {slope:.6f}c + {intercept:.6f}")
    print(f"R² = {r_squared:.6f}")
    
    # =============================================
    # Calculate Molar Absorptivity
    # =============================================
    # Check if units are in molar units for molar absorptivity calculation
    molar_units_detected = False
    normalized_units = []
    
    # Clean and normalize units for consistent detection
    for unit in units:
        # Normalize unit for consistent detection
        unit_clean = unit.lower().strip().replace('μ', 'u').replace('µ', 'u')
        normalized_units.append(unit_clean)
        
        # Check for any form of molar units
        if any(marker in unit_clean for marker in ['m', 'mol']):
            molar_units_detected = True
    
    # Debug print for unit detection
    print(f"\nDetected units: {units}")
    print(f"Normalized units for processing: {normalized_units}")
    print(f"Molar units detected: {molar_units_detected}")
    
    # If molar units detected, calculate molar absorptivity
    if molar_units_detected:
        print("\n===== Molar Absorptivity Calculation =====")
        print("Beer-Lambert Law: A = ε·c·l  where:")
        print("  ε = molar absorptivity (L·mol⁻¹·cm⁻¹)")
        print("  c = concentration (mol/L)")
        print("  l = path length (cm)")
        
        # Calculate molar absorptivity for each sample
        molar_absorptivities = []
        
        for i, row in data.iterrows():
            conc_value = row["Concentration"]
            conc_unit = row["Unit"]
            absorbance = row["Absorbance"]
            
            # Convert concentration to molar if necessary
            molar_conc, is_molar = convert_concentration_to_molar(conc_value, conc_unit)
            
            if is_molar:
                try:
                    epsilon = calculate_molar_absorptivity(absorbance, molar_conc, path_length)
                    molar_absorptivities.append(epsilon)
                    print(f"Sample {row['Sample']} ({conc_value} {conc_unit}): ε = {epsilon:.2f} L·mol⁻¹·cm⁻¹")
                except Exception as e:
                    print(f"Error calculating molar absorptivity for {row['Sample']}: {str(e)}")
                    molar_absorptivities.append(None)
            else:
                print(f"Sample {row['Sample']}: Unit {conc_unit} not recognized as molar, skipping")
                molar_absorptivities.append(None)
        
        # Add molar absorptivity to the dataframe
        data["Molar_Absorptivity"] = molar_absorptivities
        
        # Calculate average molar absorptivity (excluding None values)
        valid_epsilons = [e for e in molar_absorptivities if e is not None]
        if valid_epsilons:
            avg_epsilon = sum(valid_epsilons) / len(valid_epsilons)
            std_epsilon = np.std(valid_epsilons) if len(valid_epsilons) > 1 else 0
            print(f"\nAverage Molar Absorptivity: {avg_epsilon:.2f} ± {std_epsilon:.2f} L·mol⁻¹·cm⁻¹")
            
            # Calculate theoretical slope from Beer-Lambert law
            # Since A = ε·c·l and our calibration is A = slope·c + intercept
            # ideally, slope = ε·l
            theoretical_slope = avg_epsilon * path_length
            
            print(f"Theoretical slope from Beer-Lambert Law: {theoretical_slope:.6f}")
            print(f"Actual slope from linear regression: {slope:.6f}")
            
            if abs(theoretical_slope - slope) / theoretical_slope > 0.1:  # More than 10% difference
                print("\nNote: The difference between theoretical and actual slopes suggests possible deviations from Beer-Lambert Law.")
                print("This could be due to:") 
                print("- Concentration errors")
                print("- Instrument calibration issues")
                print("- Chemical interactions (aggregation, etc.)")
                print("- High concentration effects (inner filter effect)")
    else:
        print("\nNote: No recognized molar units (M, mM, µM, etc.) detected.")
        print("To calculate molar absorptivity, please use molar concentration units in your filenames.")
    
    # =============================================
    # Plot Results
    # =============================================
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for data points
    plt.scatter(X, y, color="red", marker="o", s=50, label="Measured Data")
    
    # Line for the fitted model
    conc_range = np.linspace(min(X) * 0.9, max(X) * 1.1, 100).reshape(-1, 1)
    plt.plot(conc_range, model.predict(conc_range), color="blue", linewidth=2, 
             label=f"Calibration Curve: A = {slope:.6f}c + {intercept:.6f}\nR² = {r_squared:.6f}")
    
    # Add data labels
    for i, (conc, abs_val, sample) in enumerate(zip(X, y, data["Sample"])):
        plt.annotate(f"{conc[0]:.2f}", 
                     (conc[0], abs_val), 
                     textcoords="offset points",
                     xytext=(0, 7), 
                     ha='center',
                     fontsize=8)
    
    plt.xlabel(concentration_label, fontsize=12)
    plt.ylabel(f"Absorbance at {target_wavelength} nm", fontsize=12)
    plt.title(f"UV-Vis Calibration Curve for {sample_label}", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Ask user if they want to save the figure
    save_choice = input("\nDo you want to save the calibration curve plot? (y/n): ").strip().lower()
    if save_choice == 'y':
        output_file = f"calibration_curve_{sample_label.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=300)
        print(f"Calibration curve saved as: {output_file}")
    
    plt.show()
    
    # =============================================
    # Save data to CSV
    # =============================================
    save_data_choice = input("\nDo you want to save all collected data to CSV? (y/n): ").strip().lower()
    if save_data_choice == 'y':
        output_csv = f"uv_vis_analysis_{sample_label.replace(' ', '_')}.csv"
        data.to_csv(output_csv, index=False)
        print(f"Data saved as: {output_csv}")
    
    # =============================================
    # Predict Unknown Sample
    # =============================================
    print(f"\nUsing calibration curve: A = {slope:.6f}c + {intercept:.6f}")
    print(f"Unit: {units[0] if len(set(units)) == 1 else 'mixed units'}")
    
    predict_choice = input("\nDo you want to predict unknown concentrations? (y/n): ").strip().lower()
    if predict_choice == 'y':
        while True:
            try:
                user_input = input("\nEnter absorbance value (or 'q' to quit): ")
                if user_input.lower() == 'q':
                    break
                    
                absorbance_value = float(user_input)
                predicted_conc = predict_concentration(absorbance_value, slope, intercept)
                
                # Check if prediction is within calibration range
                min_conc = min(concentrations)
                max_conc = max(concentrations)
                
                if predicted_conc < min_conc or predicted_conc > max_conc:
                    warning = " (Warning: outside calibration range)"
                else:
                    warning = ""
                    
                unit_display = units[0] if len(set(units)) == 1 else "mixed units"
                print(f"Predicted Concentration: {predicted_conc:.4f} {unit_display}{warning}")
                
                # If we have molar units, also calculate molar absorptivity
                if molar_units_detected and 'avg_epsilon' in locals():
                    print(f"According to Beer-Lambert Law with ε = {avg_epsilon:.2f} L·mol⁻¹·cm⁻¹:")
                    calculated_conc = absorbance_value / (avg_epsilon * path_length)
                    print(f"Calculated Concentration: {calculated_conc:.4f} M")
                
            except ValueError:
                print("Invalid input. Please enter a valid number or 'q' to quit.")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    print("\nThank you for using the UV-Vis Analysis Tool.")

if __name__ == "__main__":
    main()