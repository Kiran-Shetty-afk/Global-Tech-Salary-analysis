# Global Tech Salary Analysis

A comprehensive data analysis project exploring global data science compensation trends, featuring interactive visualizations, predictive modeling, and a web application for salary insights.

## 📋 Overview

This project analyzes the `data_science_salaries.csv` dataset to uncover patterns in tech salaries across different countries, experience levels, company sizes, and work models. It includes:

- **Exploratory Data Analysis (EDA)**: Rich visualizations and statistical summaries
- **Machine Learning Model**: Salary prediction using regression techniques
- **Interactive Web App**: Streamlit-based application for exploring salary data
- **Correlation Analysis**: Insights into factors affecting compensation

## 🚀 Features

- 12 upgraded visualizations with consistent styling
- 4 analytical summary tables
- Key insights and correlation analysis
- Machine learning evaluation with cross-validation and feature importance
- Interactive salary prediction tool
- Country-wise and role-based salary comparisons

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kiran-Shetty-afk/Global-Tech-Salary-analysis.git
   cd "DAP Project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Running the Jupyter Notebook
Open `DS_Analysis_dap.ipynb` in Jupyter Notebook or JupyterLab to explore the detailed analysis:
```bash
jupyter notebook DS_Analysis_dap.ipynb
```

### Running the Web Application
Launch the interactive Streamlit app:
```bash
streamlit run app.py
```
Navigate to the provided local URL to explore salary trends interactively.

### Using the Salary Predictor
The `salary_predictor.pkl` model can be loaded in Python for predictions:
```python
import joblib
model = joblib.load('salary_predictor.pkl')
# Use model.predict() with appropriate features
```

## 📁 Data

- **Dataset**: `data_science_salaries.csv` - Contains salary information for data science roles globally
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets) or similar (update with actual source if known)
- **Features**: Includes columns for salary, experience level, employment type, job title, company size, etc.

## 🤖 Machine Learning

The project includes a trained regression model for salary prediction based on:
- Experience level
- Company size
- Employment type
- Job title
- Remote work ratio

Model evaluation includes cross-validation, residuals analysis, and feature importance.

## 📈 Key Insights

- Analysis of salary distributions by country and role
- Impact of experience and company size on compensation
- Trends in remote vs. on-site work
- Correlation between various factors and salary

## 📝 Requirements

- Python 3.8+
- Libraries: streamlit, pandas, numpy, joblib, matplotlib, seaborn, scikit-learn

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.