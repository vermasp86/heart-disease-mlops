"""
Exploratory Data Analysis for Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plotly_template = "plotly_dark"

class HeartDiseaseEDA:
    """Class for performing EDA on Heart Disease dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize EDA class
        
        Args:
            data_path (str): Path to processed data CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        self.figures_dir = Path("notebooks/figures")
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Feature descriptions for better visualization
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia (1-3)',
            'target': 'Heart disease (1 = yes; 0 = no)',
            'source': 'Data source (cleveland, hungarian, switzerland, va)'
        }
        
        # Update column names with descriptions
        self.df = self.df.rename(columns=self.feature_descriptions)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistics"""
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        # Dataset info
        print("\nDataset Information:")
        print(f"Total samples: {len(self.df)}")
        print(f"Total features: {len(self.df.columns) - 1}")  # Excluding target
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Data types
        print("\nData Types:")
        print(self.df.dtypes)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.df.describe().T.round(2))
        
        # Class distribution
        print("\nClass Distribution:")
        class_dist = self.df['Heart disease (1 = yes; 0 = no)'].value_counts()
        class_pct = self.df['Heart disease (1 = yes; 0 = no)'].value_counts(normalize=True) * 100
        
        for idx, (count, pct) in enumerate(zip(class_dist, class_pct)):
            label = "Disease" if idx == 1 else "No Disease"
            print(f"{label}: {count} samples ({pct:.1f}%)")
    
    def visualize_class_distribution(self):
        """Visualize target class distribution"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Class Distribution', 'Percentage Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        class_counts = self.df['Heart disease (1 = yes; 0 = no)'].value_counts()
        fig.add_trace(
            go.Bar(
                x=['No Disease', 'Disease'],
                y=class_counts.values,
                marker_color=['lightblue', 'salmon'],
                text=class_counts.values,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Pie chart
        class_pct = self.df['Heart disease (1 = yes; 0 = no)'].value_counts(normalize=True) * 100
        fig.add_trace(
            go.Pie(
                labels=['No Disease', 'Disease'],
                values=class_pct.values,
                marker_colors=['lightblue', 'salmon'],
                textinfo='label+percent'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Heart Disease Class Distribution",
            showlegend=False,
            template=plotly_template
        )
        
        fig.write_html(str(self.figures_dir / "class_distribution.html"))
        fig.show()
    
    def visualize_feature_distributions(self):
        """Visualize distributions of all features"""
        numerical_features = [
            'Age in years',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Maximum heart rate achieved',
            'ST depression induced by exercise relative to rest'
        ]
        
        categorical_features = [
            'Sex (1 = male; 0 = female)',
            'Chest pain type (0-3)',
            'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'Resting electrocardiographic results (0-2)',
            'Exercise induced angina (1 = yes; 0 = no)',
            'Slope of the peak exercise ST segment (0-2)',
            'Number of major vessels (0-3) colored by fluoroscopy',
            'Thalassemia (1-3)'
        ]
        
        # Numerical features distribution
        fig_num = make_subplots(
            rows=2, cols=3,
            subplot_titles=numerical_features,
            specs=[[{"type": "histogram"}]*3, [{"type": "histogram"}]*3]
        )
        
        for idx, feature in enumerate(numerical_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            fig_num.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    nbinsx=30,
                    marker_color='lightblue',
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add vertical line for mean
            mean_val = self.df[feature].mean()
            fig_num.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                row=row, col=col
            )
        
        fig_num.update_layout(
            title_text="Numerical Features Distribution",
            height=800,
            template=plotly_template
        )
        
        fig_num.write_html(str(self.figures_dir / "numerical_distributions.html"))
        fig_num.show()
        
        # Categorical features distribution
        fig_cat = make_subplots(
            rows=3, cols=3,
            subplot_titles=categorical_features,
            specs=[[{"type": "bar"}]*3]*3
        )
        
        for idx, feature in enumerate(categorical_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            value_counts = self.df[feature].value_counts().sort_index()
            
            fig_cat.add_trace(
                go.Bar(
                    x=[str(x) for x in value_counts.index],
                    y=value_counts.values,
                    marker_color='lightgreen',
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_cat.update_layout(
            title_text="Categorical Features Distribution",
            height=1000,
            template=plotly_template
        )
        
        fig_cat.write_html(str(self.figures_dir / "categorical_distributions.html"))
        fig_cat.show()
    
    def visualize_correlation_matrix(self):
        """Create correlation matrix heatmap"""
        # Calculate correlation matrix
        corr_matrix = self.df.corr(numeric_only=True)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.mask(mask),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_matrix.mask(mask), 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            width=800,
            height=800,
            template=plotly_template,
            xaxis=dict(tickangle=45)
        )
        
        fig.write_html(str(self.figures_dir / "correlation_matrix.html"))
        fig.show()
        
        # Print top correlations with target
        target_corr = corr_matrix['Heart disease (1 = yes; 0 = no)'].abs().sort_values(ascending=False)
        print("\nTop features correlated with target:")
        for feature, corr in target_corr[1:6].items():  # Skip target itself
            print(f"{feature}: {corr:.3f}")
    
    def visualize_feature_vs_target(self):
        """Visualize relationship between features and target"""
        numerical_features = [
            'Age in years',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Maximum heart rate achieved',
            'ST depression induced by exercise relative to rest'
        ]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=numerical_features,
            specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "box"}, {"type": "scatter"}]]
        )
        
        for idx, feature in enumerate(numerical_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            if idx < 5:  # Box plots for first 5 features
                for target_val in [0, 1]:
                    target_data = self.df[self.df['Heart disease (1 = yes; 0 = no)'] == target_val][feature]
                    target_label = "No Disease" if target_val == 0 else "Disease"
                    
                    fig.add_trace(
                        go.Box(
                            y=target_data,
                            name=target_label,
                            boxpoints='outliers',
                            marker_color='lightblue' if target_val == 0 else 'salmon',
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
            else:  # Scatter plot for last feature
                fig.add_trace(
                    go.Scatter(
                        x=self.df[feature],
                        y=self.df['Heart disease (1 = yes; 0 = no)'],
                        mode='markers',
                        marker=dict(
                            color=self.df['Heart disease (1 = yes; 0 = no)'],
                            colorscale=['lightblue', 'salmon'],
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Feature Distribution by Target Class",
            height=800,
            template=plotly_template,
            boxmode='group'
        )
        
        fig.write_html(str(self.figures_dir / "feature_vs_target.html"))
        fig.show()
    
    def check_missing_values(self):
        """Check and visualize missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).sort_values('Missing Values', ascending=False)
        
        print("\nMissing Values Analysis:")
        print(missing_df[missing_df['Missing Values'] > 0])
        
        if missing.sum() > 0:
            fig = go.Figure(data=go.Bar(
                x=missing_df.index,
                y=missing_df['Percentage'],
                marker_color='coral',
                text=missing_df['Missing Values'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Missing Values Percentage by Feature",
                xaxis_title="Features",
                yaxis_title="Percentage Missing",
                template=plotly_template
            )
            
            fig.write_html(str(self.figures_dir / "missing_values.html"))
            fig.show()
        else:
            print("✓ No missing values found in the dataset.")
    
    def check_outliers(self):
        """Detect and visualize outliers using IQR method"""
        numerical_features = [
            'Age in years',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Maximum heart rate achieved',
            'ST depression induced by exercise relative to rest'
        ]
        
        outliers_info = {}
        
        for feature in numerical_features:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            outliers_pct = (len(outliers) / len(self.df)) * 100
            
            outliers_info[feature] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': len(outliers),
                'outliers_pct': outliers_pct
            }
        
        # Create summary DataFrame
        outliers_df = pd.DataFrame(outliers_info).T
        print("\nOutliers Analysis:")
        print(outliers_df.round(2))
        
        # Visualize outliers
        fig = go.Figure()
        
        for feature in numerical_features:
            fig.add_trace(go.Box(
                y=self.df[feature],
                name=feature,
                boxpoints='outliers',
                marker_color='lightblue'
            ))
        
        fig.update_layout(
            title="Outliers Detection in Numerical Features",
            yaxis_title="Feature Values",
            template=plotly_template,
            height=600
        )
        
        fig.write_html(str(self.figures_dir / "outliers_detection.html"))
        fig.show()
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("="*50)
        print("EDA REPORT SUMMARY")
        print("="*50)
        
        # Dataset overview
        print(f"\n1. DATASET OVERVIEW")
        print(f"   - Total samples: {len(self.df)}")
        print(f"   - Total features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Class balance
        class_ratio = self.df['Heart disease (1 = yes; 0 = no)'].mean()
        print(f"\n2. CLASS BALANCE")
        print(f"   - Disease prevalence: {class_ratio:.1%}")
        if 0.4 <= class_ratio <= 0.6:
            print("   - Status: ✓ Well balanced")
        else:
            print("   - Status: ⚠️ Imbalanced - consider balancing techniques")
        
        # Missing values
        missing_total = self.df.isnull().sum().sum()
        print(f"\n3. MISSING VALUES")
        print(f"   - Total missing values: {missing_total}")
        if missing_total == 0:
            print("   - Status: ✓ Complete dataset")
        else:
            print(f"   - Status: ⚠️ {missing_total} missing values found")
        
        # Data types
        numeric_count = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(self.df.select_dtypes(include=['object', 'category']).columns)
        print(f"\n4. DATA TYPES")
        print(f"   - Numerical features: {numeric_count}")
        print(f"   - Categorical features: {categorical_count}")
        
        # Key findings
        print(f"\n5. KEY FINDINGS")
        
        # Age distribution
        age_stats = self.df['Age in years'].describe()
        print(f"   - Age range: {age_stats['min']:.0f} to {age_stats['max']:.0f} years")
        print(f"   - Average age: {age_stats['mean']:.1f} years")
        
        # Cholesterol insights
        high_chol = len(self.df[self.df['Serum cholesterol (mg/dl)'] > 240])
        print(f"   - Patients with high cholesterol (>240 mg/dl): {high_chol} ({high_chol/len(self.df):.1%})")
        
        # Blood pressure insights
        high_bp = len(self.df[self.df['Resting blood pressure (mm Hg)'] > 140])
        print(f"   - Patients with high BP (>140 mmHg): {high_bp} ({high_bp/len(self.df):.1%})")
        
        # Save report to file
        report_lines = []
        report_lines.append("="*50)
        report_lines.append("HEART DISEASE DATASET - EDA REPORT")
        report_lines.append("="*50)
        report_lines.append(f"\nGenerated on: {pd.Timestamp.now()}")
        report_lines.append(f"Dataset: {self.data_path.name}")
        report_lines.append(f"Total samples: {len(self.df)}")
        report_lines.append(f"Total features: {len(self.df.columns)}")
        report_lines.append(f"Target variable: Heart disease (1 = yes; 0 = no)")
        report_lines.append(f"Class distribution: {class_ratio:.1%} positive cases")
        
        report_path = self.figures_dir / "eda_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ EDA report saved to: {report_path}")
        print(f"✓ All visualizations saved to: {self.figures_dir}/")

def main():
    """Main function to run EDA"""
    # Initialize EDA
    eda = HeartDiseaseEDA("data/processed/heart_disease_processed.csv")
    
    # Load data
    df = eda.load_data()
    
    # Perform analysis
    eda.basic_statistics()
    eda.check_missing_values()
    eda.visualize_class_distribution()
    eda.visualize_feature_distributions()
    eda.visualize_correlation_matrix()
    eda.visualize_feature_vs_target()
    eda.check_outliers()
    
    # Generate final report
    eda.generate_report()
    
    print("\n" + "="*50)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
