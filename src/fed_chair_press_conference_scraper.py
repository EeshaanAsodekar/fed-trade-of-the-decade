import pandas as pd
import requests
import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup


def download_and_extract_fomc_press_conferences(dates, output_dir='data/raw/fomc_press_conf'):
    base_url = 'https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{}.pdf'
    pdf_dir = os.path.join(output_dir, 'pdfs')
    text_dir = os.path.join(output_dir, 'texts')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    
    for date in dates:
        url = base_url.format(date)
        print(f"Attempting to download: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            pdf_filename = f'FOMCpresconf{date}.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_filename}")
            
            # Extract text from PDF
            try:
                reader = PdfReader(pdf_path)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                # Save text to file
                text_filename = f'FOMCpresconf{date}.txt'
                text_path = os.path.join(text_dir, text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted text to: {text_filename}")
            except Exception as e:
                print(f"Failed to extract text from {pdf_filename}: {e}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
    print("Done.")


def download_and_extract_fomc_meeting_minutes(dates, output_dir='data/raw/fomc_meeting_minutes'):
    base_url = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes{}.pdf'
    pdf_dir = os.path.join(output_dir, 'pdfs')
    text_dir = os.path.join(output_dir, 'texts')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    for date in dates:
        url = base_url.format(date)
        print(f"Attempting to download: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            pdf_filename = f'fomcminutes{date}.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_filename}")

            # Extract text from PDF
            try:
                reader = PdfReader(pdf_path)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                # Save text to file
                text_filename = f'fomcminutes{date}.txt'
                text_path = os.path.join(text_dir, text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted text to: {text_filename}")
            except Exception as e:
                print(f"Failed to extract text from {pdf_filename}: {e}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
    print("Done.")


def download_and_extract_fomc_statements(dates, output_dir='data/raw/fomc_meeting_statements'):
    pdf_base_url = 'https://www.federalreserve.gov/monetarypolicy/files/monetary{}a1.pdf'
    html_base_url = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm'
    pdf_dir = os.path.join(output_dir, 'pdfs')
    text_dir = os.path.join(output_dir, 'texts')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    for date in dates:
        pdf_url = pdf_base_url.format(date)
        print(f"Attempting to download PDF: {pdf_url}")
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_filename = f'monetary{date}a1.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_filename}")

            # Extract text from PDF
            try:
                reader = PdfReader(pdf_path)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                # Save text to file
                text_filename = f'monetary{date}a1.txt'
                text_path = os.path.join(text_dir, text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted text to: {text_filename}")
            except Exception as e:
                print(f"Failed to extract text from {pdf_filename}: {e}")
        else:
            print(f"Failed to download PDF: {pdf_url} (Status code: {response.status_code})")
            # Try downloading the HTML version
            html_url = html_base_url.format(date)
            print(f"Attempting to download HTML: {html_url}")
            html_response = requests.get(html_url)
            if html_response.status_code == 200:
                try:
                    soup = BeautifulSoup(html_response.text, 'html.parser')
                    full_text = soup.get_text(separator='\n')
                    
                    # Extract relevant portion of the text
                    start = full_text.find("For immediate release")
                    end = full_text.find("Voting for the FOMC monetary policy action were")
                    if start != -1 and end != -1:
                        statement_text = full_text[start:end].strip()
                        # Remove unnecessary whitespace
                        statement_text = '\n'.join(line.strip() for line in statement_text.splitlines() if line.strip())
                        
                        # Save the cleaned text to file
                        text_filename = f'monetary{date}a1.txt'
                        text_path = os.path.join(text_dir, text_filename)
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(statement_text)
                        print(f"Extracted and saved relevant text to: {text_filename}")
                    else:
                        print(f"Could not find the relevant section in the HTML for {date}.")
                except Exception as e:
                    print(f"Failed to extract text from HTML for {date}: {e}")
            else:
                print(f"Failed to download HTML: {html_url} (Status code: {html_response.status_code})")
    print("Done.")


def save_fomc_meeting_dates(dates, output_file="data/processed/fomc_meeting_dates.csv"):
    """
    Takes a list of FOMC meeting dates in 'YYYYMMDD' format, converts them to a pandas datetime column,
    and saves them to a CSV file.

    Args:
        dates (list): List of dates in 'YYYYMMDD' format.
        output_file (str): Output file name for the CSV.

    Returns:
        None
    """
    # Create a DataFrame from the dates
    df = pd.DataFrame({"date": dates})
    
    # Convert the 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved FOMC meeting dates to {output_file}.")


if __name__ == '__main__':
    # NOTE: not all fomc meetings are followed by press conferences by the Fed Chair
    dates = [
        # 2012
        '20120125', '20120313', '20120425', '20120620', '20120801', '20120913', '20121024','20121212',
        # 2013
        '20130130', '20130320', '20130501', '20130619', '20130731', '20130918', '20131030','20131218',
        # 2014
        '20140129', '20140319', '20140430', '20140618', '20140730', '20140917', '20141029', '20141217', 
        # 2015
        '20150128', '20150318', '20150429', '20150617', '20150729', '20150917', '20151028', '20151216',
        # 2016
        '20160127', '20160316', '20160427', '20160615', '20160727', '20160921', '20161102', '20161214',
        # 2017
        '20170201', '20170315', '20170503', '20170614', '20170726', '20170920', '20171101', '20171213',
        # 2018
        '20180131', '20180321', '20180502', '20180613', '20180801', '20180926', '20181108', '20181219',
        # 2019
        '20190130', '20190320', '20190501', '20190619', '20190731', '20190918', '20191030', '20191211',
        # 2020
        '20200129', '20200303', '20200315', '20200429', '20200610', '20200729', '20200916', '20201105', '20201216',
        # 2021
        '20210127', '20210317', '20210428', '20210616', '20210728', '20210922', '20211103', '20211215',
        # 2022
        '20220126', '20220316', '20220504', '20220615', '20220727', '20220921', '20221102', '20221214',
        # 2023
        '20230201', '20230322', '20230503', '20230614', '20230726', '20230920', '20231101', '20231213',
        # 2024
        '20240131', '20240320', '20240501', '20240612', '20240731', '20240918', '20241107', 
        # TODO: Need to update with latest dates for data pull

    ]

    
    save_fomc_meeting_dates(dates)
    download_and_extract_fomc_press_conferences(dates)
    download_and_extract_fomc_statements(dates)
    download_and_extract_fomc_meeting_minutes(dates)
