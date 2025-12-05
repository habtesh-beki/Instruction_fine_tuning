import json
import requests

Data_path = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
Val_Rate = 0.1
Test_Rate = 0.06

try:
    # Make an HTTP GET request to the URL
    response = requests.get(Data_path)

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # Get the JSON data from the response
    # requests automatically handles JSON decoding if the Content-Type header is correct
    data = response.json()
    output_filename = "downloaded_data.json" # Define the filename for saving the JSON data
    with open(output_filename, "w", encoding="utf-8") as f:  # Write the JSON data to a local file
        json.dump(data, f, indent=4)  # Use indent for pretty-printing the JSON

    print(f"JSON data successfully downloaded and saved to '{output_filename[:10]}'")

except requests.exceptions.RequestException as e:
    print(f"Error downloading JSON file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")