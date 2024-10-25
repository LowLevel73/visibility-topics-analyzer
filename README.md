
# Visibility Topics Analyzer

**Version**: 0.9.0  
**Author**: Enrico Altavilla

## Overview

**Visibility Topics Analyzer** is a Python script that processes Google Search Console (GSC) data to analyze visibility changes of a website for different search queries over two dates. The script identifies and ranks the most significant topics associated with visibility changes using **Non-negative Matrix Factorization (NMF)**. The results help you understand which topics were positively or negatively impacted by the ranking changes.

## Features

- Reads visibility data (search queries and their positions) from an Excel file.
- Cleans and processes the data, including removing duplicates and handling missing values.
- Uses **NMF** to extract latent topics from search queries.
- Identifies the most impactful topics related to ranking changes.
- Outputs a bar chart using **Plotly** to visually present the results.

## Usage

1. Prepare your visibility data in an Excel file with at least three columns:
   - Search queries
   - New positions (more recent data)
   - Old positions (less recent data)
   
   The order of the columns can be specified when running the script.

2. Run the script:

   ```bash
   python visibility-topics-analyzer.py <file_name>
   ```

   Replace `<file_name>` with the path to your Excel file.

3. The script will prompt you to specify which columns in your file contain search queries and position data, if needed.

4. The output is an interactive Plotly bar chart that shows how each topic was associated with ranking changes.

## Example

```bash
python visibility-topics-analyzer.py data/visibility_data.xlsx
```

## Dependencies

The script relies on the following Python libraries:

- **sys** (built-in)
- **time** (built-in)
- **numpy**
- **pandas**
- **scikit-learn** (`TfidfVectorizer`, `LinearRegression`, `NMF`, `StandardScaler`)
- **plotly**
- **nltk** (for stopword filtering)
  
You can install the required libraries using:

```bash
pip install numpy pandas scikit-learn plotly nltk
```

### Data Format

- Ensure that your Excel file has three columns: 
   - **Search Queries**: The terms or phrases that users searched for.
   - **New Positions**: The average position of the search queries on the most recent date.
   - **Old Positions**: The average position of the search queries on a previous date.

The script calculates the difference between the two position columns to identify the change in rankings.

### Stopwords

The script supports stopword removal for different languages. Currently, it defaults to **Italian stopwords**. You can modify this for other languages in the script.

## License

This project is licensed under the MIT License.
