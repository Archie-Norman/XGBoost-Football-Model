import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
import time

# Set up the WebDriver with headed mode
options = Options()
options.headless = False  # Use headed mode for visibility

# Initialize the WebDriver with options
driver = webdriver.Chrome(options=options)

# Open the main website
website_url = "https://www.betfair.com/betting/football/ac-1"
driver.get(website_url)

# Wait for the parent element containing the links to load
parent_xpath = "/html/body/div/div[2]/div[2]/div/div/section/div[1]/div/div[2]/div[1]/div/div/div/div"

WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, parent_xpath)))

# Locate all links within the parent XPath
links_xpath = f"{parent_xpath}//a[@href]"
link_elements = driver.find_elements(By.XPATH, links_xpath)

# Extract URLs from the links
all_links = [link_element.get_attribute("href") for link_element in link_elements]

# Filter links that start with "https://www.betfair.com/betting/football"
links = [link for link in all_links if link.startswith("https://www.betfair.com/betting/football")]


# Target XPath for extracting text from the children
text_target_xpath = '/html/body/div/div[2]/div[2]/div/div/section/div[1]/div/div/div/div[2]/div/div/div[2]/div/div/div[4]/div'

# Prepare to save data to a CSV file
output_file = "extracted_text_with_children.csv"

# Open the CSV file for writing
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["URL", "Child Index", "Child Text"])
    
    # Visit each link and extract the desired text
    for index, link in enumerate(links):
        try:
            print(f"Processing link {index + 1}: {link}")
            
            # Open the link in the browser
            driver.get(link)
            
            # Wait for the text container to load
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, text_target_xpath)))
            
            # Locate the parent element containing the children
            parent_element = driver.find_element(By.XPATH, text_target_xpath)

            # Scroll the element into view
            driver.execute_script("arguments[0].scrollIntoView();", parent_element)
            time.sleep(2)  # Small delay to allow loading

            # Scroll down multiple times to load all elements
            for _ in range(5):  # Adjust the range as needed
                driver.execute_script("arguments[0].scrollTop += 300;", parent_element)
                time.sleep(1)  # Allow time for loading
            
            # Locate all child elements within the parent
            child_elements = parent_element.find_elements(By.XPATH, "./*")
            
            # Iterate through each child element and extract text
            for child_index, child in enumerate(child_elements, start=1):
                child_text = child.text.strip()  # Extract and clean the text
                print(f"Extracted text from child {child_index}: {child_text}")
                
                # Write the URL, child index, and child text to the CSV file
                writer.writerow([link, child_index, child_text])
        
        except Exception as e:
            print(f"Error processing link {index + 1}: {e}")
            # Write an error message to the CSV file for debugging
            writer.writerow([link, "Error", str(e)])

# Close the WebDriver
driver.quit()


import pandas as pd
import re
from datetime import datetime

# Read the CSV into a DataFrame
df = pd.read_csv(output_file)

# Drop rows with Child Index 1
df = df[df['Child Index'] != 1]

# Function to convert fractional odds to decimal odds
def fractional_to_decimal(fractional_odds):
    if '/' in fractional_odds:
        numerator, denominator = map(int, fractional_odds.split('/'))
        return (numerator / denominator) + 1
    return None

# Function to extract home, away, date, time, and odds from the Child Text
def extract_info(text):
    if isinstance(text, str):
        lines = text.split("\n")
        
        if len(lines) >= 4:
            home = lines[0].strip()
            away = lines[1].strip()
            date = lines[2].strip()
            time = lines[3].strip()
        else:
            return None
        
        if not re.match(r'\d{1,2} \w{3}', date) or not re.match(r'\d{2}:\d{2}', time):
            return None  
        
        odds = re.findall(r'(\d+/\d+)', text)
        
        if len(odds) >= 3:
            home_odds = fractional_to_decimal(odds[0])
            draw_odds = fractional_to_decimal(odds[1])
            away_odds = fractional_to_decimal(odds[2])
        else:
            return None

        return {
            'Home': home,
            'Away': away,
            'Date': date,
            'Time': time,
            'Home Odds': home_odds,
            'Draw Odds': draw_odds,
            'Away Odds': away_odds,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add timestamp
        }
    return None

# Apply the extraction function to each row of the 'Child Text' column
extracted_data = df['Child Text'].apply(extract_info)

# Filter out None values (if any)
extracted_data = extracted_data.dropna()

# Convert extracted data into a DataFrame
extracted_df = pd.DataFrame(extracted_data.tolist())

# **Append** the extracted data to the existing CSV instead of overwriting it
output_file = "processed_extracted_data.csv"
extracted_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False, encoding="utf-8")

# Display the result (optional)
print(extracted_df)

