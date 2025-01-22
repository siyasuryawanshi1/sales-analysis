
# Sales Anlaysis Dashboard

The Sales Analysis Dashboard is a Streamlit web application designed to analyze sales leads data. It provides key insights such as win rates, average process duration, and active leads, along with detailed visualizations to enhance decision-making.
Check it out here 🚀: https://sales-analysis-6q8n4fbauod5eeazt4pvzp.streamlit.app/ 

### Features
- Upload CSV files to analyze sales leads and their performance.
- **Interactive visualizations**:
    - Lead state distribution
    - Sales funnel analysis
    - Asignee Performance metrics
- **Duration Analysis** : Detailed stage duration breakdown for different sales stages.
- **Key Metrics** : Total leads, Win rates, Average process duration, Active lead counts. 


### Installation
1. Clone the repository
```bash
git clone https://github.com/siyasuryawanshi1/sales-analysis.git
cd sales-analysis
```
2. Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage
1. Start the streamlit server
```bash
streamlit run app.py
```
2. Open the provided URL in your browser (usually http://localhost:8501).
3. Upload your CSV file to analyze the sales data.


### Data Requirements

The uploaded CSV file should contain the following columns:
- Created Date
- Next Action Date
- DemoDate
- QuoteSentDate
- Order Status
- Additional relevant fields for analysis
Ensure date columns are in the dd/mm/yy format.


### Acknowledgements
**Libraries Used**: Streamlit, Pandas, Plotly, Lifelines


## License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the LICENSE file for details.


