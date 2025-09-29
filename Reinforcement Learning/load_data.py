import pandas as pd
from azure.storage.filedatalake import DataLakeFileClient

# --- credential ---
CONNECTION_STRING = "your_connection_string_goes_here"
FILE_SYSTEM_NAME = "name_of_the_file_system_or_container"
FILE_PATH = "path/to/your/datafile.csv" 
#----------------------------------------------------

try:
    # 1. Create a client to connect to the specific file
    file_client = DataLakeFileClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        file_system_name=FILE_SYSTEM_NAME,
        file_path=FILE_PATH
    )

    # 2. Download the file's content
    download = file_client.download_file()
    file_contents = download.readall()

    # 3. Load the data into pandas
    # This example assumes a CSV file. If it's Parquet, you'd use pd.read_parquet()
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_contents))

    print("Successfully downloaded and loaded data!")
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")