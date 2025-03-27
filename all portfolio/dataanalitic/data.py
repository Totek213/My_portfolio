import os
import pandas as pd

def load_and_analyze_file(file_path):
    """Wczytuje i analizuje dane z pojedynczego pliku."""
    try:
        # Ładowanie pliku w zależności od formatu
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            return f"Unsupported file type: {file_path}"

        # Analiza danych
        info = {
            'File': file_path,
            'Rows': len(data),
            'Columns': len(data.columns),
            'Columns Info': data.dtypes.to_dict(),
            'Missing Data': data.isnull().sum().to_dict(),
            'Sample Data': data.head().to_dict(orient='records')  # Przykład pierwszych 5 wierszy
        }
        return info

    except Exception as e:
        return f"Error loading {file_path}: {str(e)}"

def analyze_data_in_folder(folder_path):
    """Analizuje dane z wszystkich plików w folderze."""
    analysis_report = []
    supported_extensions = {'.csv', '.xlsx', '.json'}  # Obsługiwane formaty
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Sprawdzanie czy to plik i czy ma obsługiwane rozszerzenie
        if os.path.isfile(file_path):
            extension = os.path.splitext(file_path)[1].lower()  # Pobiera rozszerzenie pliku
            if extension in supported_extensions:
                result = load_and_analyze_file(file_path)
                analysis_report.append(result)
            else:
                print(f"Skipped unsupported file: {file_path}")

    return analysis_report

# Ścieżka do folderu z plikami
folder_path = r"C:\Users\Admin\OneDrive\Pulpit"  # Zmień na swoją ścieżkę

# Uruchomienie analizy
report = analyze_data_in_folder(folder_path)

# Wyświetlenie raportu
for item in report:
    print(item)
