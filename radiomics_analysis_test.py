import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk  # Ensure you have the Pillow library installed
from lifelines import CoxPHFitter
import os
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)



class RadiomicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radiomics Analysis Software")
        rights_notice = tk.Label(
        self.root,
        text="All rights belong to the Mi*Edge consortium, Grant No. 031L0237C (MiEDGE project/ERACOSYSMED)",
        font=("Arial", 12, "italic"),
        fg="black",
        anchor="w",  # Align to the left
        justify="left",  # Left-align the text
        wraplength=800  # Wrap text to fit within the window
        )
        rights_notice.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=5)
        # Add a logo frame at the top-right corner
        logo_frame = tk.Frame(self.root, bg="white")  # Frame for the logo
        logo_frame.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=10)  # Top-right position

        # Set the size of the GUI
        self.root.geometry("900x700")  # Larger window size for better UI

        # Load the logo image
        try:
            logo_image = Image.open(r"C:\Users\pejma\Nextcloud\BaM_Model\logo.png")  # Replace with your logo file path
            logo_image = logo_image.resize((70, 70), Image.Resampling.LANCZOS)  # Use Resampling.LANCZOS instead of ANTIALIAS
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(logo_frame, image=logo_photo, bg="white")
            logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            logo_label.pack()
        except Exception as e:
            print(f"Error loading logo: {e}")
        # Input file section
        input_frame = tk.LabelFrame(root, text="Input File", font=("Arial", 12), padx=10, pady=10)
        input_frame.pack(pady=10, fill="x")
        
        self.file_label = tk.Label(input_frame, text="Select Radiomics Data File:", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=5)
        self.file_button = tk.Button(input_frame, text="Browse", font=("Arial", 12), command=self.load_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        # Feature selection section
        feature_frame = tk.LabelFrame(root, text="Feature Selection Settings", font=("Arial", 12), padx=10, pady=10)
        feature_frame.pack(pady=10, fill="x")
        
        self.var_threshold_label = tk.Label(feature_frame, text="Variance Threshold (default: 0.01):", font=("Arial", 12))
        self.var_threshold_label.pack(pady=5)
        self.var_threshold_entry = tk.Entry(feature_frame, font=("Arial", 12))
        self.var_threshold_entry.insert(0, "0.01")
        self.var_threshold_entry.pack(pady=5)
        
        self.corr_threshold_label = tk.Label(feature_frame, text="Correlation Threshold (default: 0.9):", font=("Arial", 12))
        self.corr_threshold_label.pack(pady=5)
        self.corr_threshold_entry = tk.Entry(feature_frame, font=("Arial", 12))
        self.corr_threshold_entry.insert(0, "0.9")
        self.corr_threshold_entry.pack(pady=5)
        
        self.additional_features_frame = tk.LabelFrame(feature_frame, text="Additional Features to Drop", font=("Arial", 12), padx=10, pady=10)
        self.additional_features_frame.pack(pady=10, fill="x")
        self.additional_features_vars = {}

        # Metric selection section
        metric_frame = tk.LabelFrame(root, text="Metric Selection", font=("Arial", 12), padx=10, pady=10)
        metric_frame.pack(pady=10, fill="x")
        
        self.metric_label = tk.Label(metric_frame, text="Select Metric for Analysis:", font=("Arial", 12))
        self.metric_label.pack(pady=5)
        self.metric_selection = ttk.Combobox(metric_frame, values=["Feature Importance", "Mutual Information"], font=("Arial", 12))
        self.metric_selection.current(0)  # Default to Feature Importance
        self.metric_selection.pack(pady=5)
        
        # Action buttons with grid layout
        action_frame = tk.LabelFrame(root, text="Actions", font=("Arial", 12), padx=10, pady=10)
        action_frame.pack(pady=(10, 20), fill="x")  # Keep the frame at the bottom

        # Define a fixed grid for action buttons
        self.process_button = tk.Button(action_frame, text="Process Data", font=("Arial", 12), command=self.process_data)
        self.process_button.grid(row=0, column=0, padx=10, pady=5)

        self.results_button = tk.Button(action_frame, text="Show Results", font=("Arial", 12), command=self.show_results, state=tk.DISABLED)
        self.results_button.grid(row=0, column=1, padx=10, pady=5)

        self.scatter_button = tk.Button(action_frame, text="Show Scatter Plot", font=("Arial", 12), command=self.show_scatter_plot, state=tk.DISABLED)
        self.scatter_button.grid(row=0, column=2, padx=10, pady=5)

        self.heatmap_button = tk.Button(action_frame, text="Show Heatmap", font=("Arial", 12), command=self.show_heatmap, state=tk.DISABLED)
        self.heatmap_button.grid(row=0, column=3, padx=10, pady=5)

        self.save_button = tk.Button(action_frame, text="Save Processed Data", font=("Arial", 12), command=self.save_data, state=tk.DISABLED)
        self.save_button.grid(row=0, column=4, padx=10, pady=5)
        
        
        # Statistical Analysis Section
        stat_frame = tk.LabelFrame(root, text="Statistical Analysis", font=("Arial", 12), padx=10, pady=10)
        stat_frame.pack(pady=10, fill="x")

        # Cox Regression Button
        self.cox_button = tk.Button(stat_frame, text="Perform Cox Regression", font=("Arial", 12), command=self.perform_cox_regression)
        self.cox_button.pack(side=tk.LEFT, padx=10, pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.populate_additional_features()
                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def populate_additional_features(self):
        for widget in self.additional_features_frame.winfo_children():
            widget.destroy()

        potential_features = ['Age', 'Sex', 'BMI', 'Other_Metadata']
        existing_features = [col for col in potential_features if col in self.data.columns]

        if not existing_features:
            tk.Label(self.additional_features_frame, text="No additional features detected.", font=("Arial", 12)).pack()
        else:
            self.additional_features_vars = {}
            for feature in existing_features:
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(
                    self.additional_features_frame, text=feature, font=("Arial", 12), variable=var
                )
                checkbox.pack(anchor="w")
                self.additional_features_vars[feature] = var

    def process_data(self):
        try:
            var_threshold = float(self.var_threshold_entry.get())
            corr_threshold = float(self.corr_threshold_entry.get())
            features_to_drop = ['patient', 'survival_time']
            for feature, var in self.additional_features_vars.items():
                if var.get():
                    features_to_drop.append(feature)

            radiomics_features = self.data.drop(columns=features_to_drop, errors='ignore')

            if radiomics_features.empty:
                raise ValueError("No radiomics features found after excluding additional features!")

            selector = VarianceThreshold(threshold=var_threshold)
            reduced_features = selector.fit_transform(radiomics_features)
            self.reduced_features_df = pd.DataFrame(
                reduced_features,
                columns=radiomics_features.columns[selector.get_support()]
            )
            
            corr_matrix = self.reduced_features_df.corr()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
            self.final_features_df = self.reduced_features_df.drop(columns=to_drop)

            self.data['Survival_Category'] = self.data['survival_time'].apply(
                lambda x: 'Poor' if x / 30 <= 15 else 'Good'
            )
            label_encoder = LabelEncoder()
            self.data['Survival_Category_Numeric'] = label_encoder.fit_transform(self.data['Survival_Category'])
            self.target = self.data['Survival_Category_Numeric']
            self.features = self.final_features_df

            X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
            self.rf = RandomForestClassifier(random_state=42)
            self.rf.fit(X_train, y_train)
            self.test_accuracy = self.rf.score(X_test, y_test)

            self.results_button.config(state=tk.NORMAL)
            self.scatter_button.config(state=tk.NORMAL)
            self.heatmap_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", "Data processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process data: {e}")

    def show_results(self):
        try:
            # Get the selected metric
            selected_metric = self.metric_selection.get()
            self.selected_metric = selected_metric  # Store the selected metric for other functions
            
            # Clear any previous results
            self.top_features = None

            if selected_metric == "Feature Importance":
                # Compute and sort feature importance
                importance = self.rf.feature_importances_
                sorted_idx = np.argsort(importance)[::-1]
                self.top_features = self.features.columns[sorted_idx[:20]]
                
                # Plot feature importance
                plt.figure(figsize=(16, 8))
                plt.bar(range(20), importance[sorted_idx[:20]], alpha=0.7)
                plt.xticks(range(20), self.top_features, rotation=45, ha="right", fontsize=10)
                plt.title('Top 20 Feature Importance', fontsize=14)
                plt.ylabel('Importance', fontsize=12)
                plt.xlabel('Features', fontsize=12)
                plt.tight_layout()
                plt.show()
            
            elif selected_metric == "Mutual Information":
                # Compute and sort mutual information
                mi_scores = mutual_info_classif(self.features, self.target, random_state=42)
                self.mi_scores_df = pd.DataFrame({'Feature': self.features.columns, 'Mutual Information': mi_scores})
                self.mi_scores_df = self.mi_scores_df.sort_values(by='Mutual Information', ascending=False).head(20)
                self.top_features = self.mi_scores_df['Feature']
                
                # Plot mutual information scores
                plt.figure(figsize=(16, 8))
                plt.barh(self.mi_scores_df['Feature'], self.mi_scores_df['Mutual Information'], alpha=0.7)
                plt.title('Mutual Information Scores', fontsize=14)
                plt.xlabel('Mutual Information', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()

            # Enable buttons for further analysis
            self.scatter_button.config(state=tk.NORMAL)
            self.heatmap_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {e}")


    def show_scatter_plot(self):
        try:
            # Get the top-ranked feature
            top_feature = self.top_features.iloc[0] if self.selected_metric == "Mutual Information" else self.top_features[0]
            
            # Normalize the feature
            scaler = MinMaxScaler()
            self.data[f'{top_feature}_normalized'] = scaler.fit_transform(self.data[[top_feature]])
            
            # Convert survival time to months
            self.data['survival_time_months'] = self.data['survival_time'] / 30
            
            # Prepare data for regression
            X = self.data[f'{top_feature}_normalized'].values.reshape(-1, 1)
            y = self.data['survival_time_months']
            
            # Fit a linear regression model
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            reg = LinearRegression()
            reg.fit(X, y)
            y_pred = reg.predict(X)
            
            # Calculate R^2
            r2 = r2_score(y, y_pred)
            
            # Plot the scatter plot with regression line
            plt.figure(figsize=(10, 6))
            plt.scatter(
                self.data[f'{top_feature}_normalized'], 
                self.data['survival_time_months'], 
                alpha=0.6, 
                color='blue', 
                label='Data Points'
            )
            plt.plot(
                self.data[f'{top_feature}_normalized'], 
                y_pred, 
                color='red', 
                linewidth=2, 
                label=f'Regression Line ($R^2 = {r2:.2f}$)'
            )
            plt.title(f'{top_feature} vs Survival Time', fontsize=14)
            plt.xlabel(f'Normalized {top_feature}', fontsize=12)
            plt.ylabel('Survival Time (months)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display scatter plot: {e}")


    def show_heatmap(self):
        try:
            # Prepare data for the heatmap
            self.data['survival_time_months'] = self.data['survival_time'] / 30
            top_features = list(self.top_features)  # Convert top features to a list
            heatmap_data = self.data[top_features + ['survival_time_months']]
            correlation_matrix = heatmap_data.corr()

            # Adjust the figure size dynamically based on the number of features
            num_features = len(top_features) + 1  # Include survival_time_months
            figsize = (min(20, num_features), min(20, num_features))  # Limit max figure size to 20x20

            plt.figure(figsize=figsize)
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                cbar=True,
                annot_kws={"size": 8},  # Adjust annotation font size
            )
            plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-axis labels for readability
            plt.yticks(fontsize=10)  # Adjust y-axis label font size
            plt.title('Heatmap of Top Features and Survival Time', fontsize=14)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display heatmap: {e}")
    def perform_cox_regression(self):
        try:
            self.data['survival_time_months'] = self.data['survival_time'] / 30
            # Prepare data for Cox regression
            cox_data = self.data[['survival_time_months', 'Survival_Category_Numeric'] + list(self.top_features)]
            cox_data = cox_data.dropna()  # Remove rows with missing values

            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='survival_time_months', event_col='Survival_Category_Numeric')

            # Get summary as DataFrame
            summary = cph.summary.round(4)  # Round values to 4 decimal places

            # Ask user where to save the file
            save_path = filedialog.asksaveasfilename(
                title="Save Cox Regression Results",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            )
            if save_path:
                # Save to CSV
                summary.to_csv(save_path)

                # Automatically open the file in Excel
                try:
                    os.startfile(save_path)  # Windows-specific
                except AttributeError:
                    messagebox.showinfo(
                        "Saved", "Results saved. Open the file manually using a spreadsheet application."
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform Cox regression: {e}")
                

    def save_data(self):
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Processed Data",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if save_path:
                self.final_features_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", "Data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RadiomicsApp(root)
    root.mainloop()
