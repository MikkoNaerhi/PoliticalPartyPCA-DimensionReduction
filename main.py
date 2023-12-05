import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# -------------------------------------------
# -------------- PREPROCESSING --------------
# -------------------------------------------

def get_file_path(filename:str):
    return os.path.join(DATA_DIR, filename)

def load_data(expert_file_name:str, party_file_name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV files into pandas DataFrames.

    Parameters:
    -----------
        expert_file_name (str): The name of the CSV file containing expert data.
        party_file_name (str): The name of the CSV file containing party data.

    Returns:
    --------
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames, the first for expert data and the second for party data.
    """
    expert_raw_data = pd.read_csv(get_file_path(expert_file_name))
    party_raw_data = pd.read_csv(get_file_path(party_file_name))

    return (expert_raw_data, party_raw_data)

def plot_missing_data_per_column(raw_data:pd.DataFrame) -> None:
    """
    Visualize the percentage of missing data in each column of the provided DataFrame.

    Parameters:
    -----------
        raw_data (pd.DataFrame): The DataFrame to analyze for missing data.

    Returns:
    --------
        None
    """
    missing_data = pd.DataFrame(raw_data.isnull().sum() / len(raw_data) * 100, columns=['% Missing'])

    plt.figure(figsize=(12, 6))
    plt.title("% of missing data in the columns")
    plt.scatter(raw_data.columns, missing_data['% Missing'])
    plt.xticks(rotation=90)
    plt.xlabel("Column Name")
    plt.ylabel("% Missing Data")
    plt.tight_layout()
    plt.show()

def preprocess_expert_data(expert_data:pd.DataFrame, impute_data:bool=True, remove_d:bool=False, save_data:bool=False) -> pd.DataFrame:
    """
    Preprocess the expert data by cleaning, handling special values, and imputing missing values.

    Parameters:
    -----------
        expert_data (pd.DataFrame): The DataFrame containing expert data.
        impute_data (bool): If True, rows containing NaN values will be mean imputed. Defaults to True.
        remove_d (bool): If True, rows containing '.d' values will be removed. Defaults to False.
        save_data (bool): If True, the cleaned data will be saved to a CSV file. Defaults to False.

    Returns:
    --------
        pd.DataFrame: The cleaned and preprocessed expert data.
    """

    # Identify columns with string values
    str_columns = ['party_name','cname']

    # Columns to be removed
    remove_cols = ['eu_position_sd', 'lrecon_sd', 'galtan_sd','party','dob','id','gender','galtan_self','lrecon_self']

    # Remove the string columns from the DataFrame
    df_numerical = expert_data.drop(columns=str_columns)

    # Remove rows containing '.d', if specified
    if remove_d:
        df_numerical = df_numerical[~(df_numerical == '.d').any(axis=1)]

    # Convert all columns to numeric, coercing errors
    df_cleaned_numeric = df_numerical.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # Drop the specified columns in 'remove_cols'
    df_cleaned_numeric.drop(columns=remove_cols, inplace=True)

    # Perform mean imputation for numerical columns
    if impute_data:
        df_cleaned_numeric = df_cleaned_numeric.fillna(df_cleaned_numeric.mean())

    # Optionally save the cleaned data
    if save_data:
        df_cleaned_numeric.to_csv(get_file_path('cleaned_data.csv'))

    # Set 'party_id' as the index
    df_cleaned_numeric.set_index('party_id',inplace=True)

    return df_cleaned_numeric

def preprocess_party_data(party_data:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the party data by selecting relevant columns.

    Parameters:
    -----------
        party_data (pd.DataFrame): The original party data.

    Returns:
    --------
        pd.DataFrame: The preprocessed party data with selected columns.
    """
    # Keep only relevant columns from party datafile
    party_data = party_data[['country','party','party_id']]
    party_data.set_index('party_id', inplace=True)

    return party_data

def aggregate_pol_parties(expert_data:pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate expert data grouped by political party. Impute missing values with mean imputation

    Parameters:
    -----------
        expert_data (pd.DataFrame): The preprocessed DataFrame containing expert data.

    Returns:
    --------
        pd.DataFrame: The aggregated expert data grouped by 'party_id'.
    """
    # Group the data together per political party
    expert_data = expert_data.reset_index()
    aggregated_data = expert_data.groupby('party_id').mean()

    # Mean imputation after aggregation
    aggregated_data = aggregated_data.fillna(aggregated_data.mean())
    return aggregated_data 

# -------------------------------------------
# -------------- UTILITY FUNCS --------------
# -------------------------------------------

def combine_data(scaled_data:pd.DataFrame, party_data:pd.DataFrame) -> pd.DataFrame:
    """
    Combine expert and party data into a single DataFrame.

    Parameters:
    -----------
        scaled_data (pd.DataFrame): The scaled expert data.
        party_data (pd.DataFrame): The preprocessed party data.

    Returns:
    --------
        pd.DataFrame: The combined DataFrame with both expert and party data.
    """
    combined_data = scaled_data.join(party_data)
    return combined_data

def generate_binary_combinations(num_features:int, num_samples:int) -> np.ndarray:
    """
    Generate a set of binary combinations for the given number of features.
    Each combination represents a scenario where each feature is at one of two extremes.
    
    Parameters:
    -----------
        num_features (int): The number of features in the dataset.
        num_samples (int): The number of binary combinations to generate.
    
    Returns:
    --------
        np.array: An array of binary combinations.
    """
    # Generate random binary combinations
    return np.random.randint(2, size=(num_samples, num_features))

# -------------------------------------------
# ---------------- MODELLING ----------------
# -------------------------------------------

def scale_data(data:pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale the provided data using StandardScaler.

    Parameters:
    -----------
        data (pd.DataFrame): The data to be scaled.

    Returns:
    --------
        tuple[pd.DataFrame, StandardScaler]: A tuple containing the scaled DataFrame and the scaler used.
    """
    scaler = StandardScaler()
    data = data.reset_index()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(index=data.index, data=scaled_data, columns=data.columns)
    scaled_df.index = data['party_id']

    return (scaled_df, scaler)

def perform_PCA(scaled_data:pd.DataFrame) -> tuple[pd.DataFrame, PCA, list]:
    """
    Perform Principal Component Analysis (PCA) on the scaled data.

    Parameters:
    -----------
        scaled_data (pd.DataFrame): The scaled data to apply PCA on.

    Returns:
    --------
        tuple[pd.DataFrame, PCA, list]: A tuple with the PCA transformed data, the PCA model, and the names of the original features.
    """
    
    features = scaled_data.drop(columns=['country','party'])
    pca_model = PCA(n_components=2)
    projections = pca_model.fit_transform(X=features)
    explained_variance = pca_model.explained_variance_ratio_

    print(f"Combined explained variance of the PCAs: {100*round(sum(explained_variance),3)} %.")

    pca_2d_data = pd.DataFrame(index=scaled_data.index,data=projections, columns=['PCA1','PCA2'])
    pca_2d_data["country"] = scaled_data["country"]
    pca_2d_data["party"] = scaled_data["party"]
    pca_2d_data['party_id'] = scaled_data.index

    return (pca_2d_data, pca_model, features.columns)

def plot_PCA(X_pca:pd.DataFrame):
    """
    Plot the results of PCA in a two-dimensional scatter plot.

    Parameters:
    -----------
        X_pca (pd.DataFrame): The PCA-transformed data to plot.

    Returns:
    --------
        None
    """
    X_pca['party_id'] = X_pca['party_id'].astype(str)
    fig = px.scatter(X_pca, 
                        x='PCA1', 
                        y= 'PCA2',
                        color="party_id",
                        opacity=0.5,
                        title="PCA Results: Two-dimensional Representation of Political Parties")
    
    fig.show()

def generate_new_samples(PCA_df:pd.DataFrame, plot_3D_dist:bool=False, plot_new_samples:bool=False):
    """
    Generate new samples from the PCA-transformed data using Gaussian Kernel Density Estimation.

    Parameters:
    -----------
        PCA_df (pd.DataFrame): The PCA-transformed data.

    Returns:
    --------
        ndarray: New samples generated from the PCA data distribution.
    """
    PCA_data = PCA_df[['PCA1','PCA2']]

    # Estimate the distribution of the PCA 2d results.
    kde = gaussian_kde(PCA_data.T)

    # Create a meshgrid for the plot
    x, y = np.mgrid[-10:10, -10:10]
    pos = np.vstack([x.ravel(), y.ravel()])

    # Evaluate the KDE model on the grid
    z = np.reshape(kde(pos).T, x.shape)

    # Create a 3D plot
    if plot_3D_dist:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot a 3D surface
        ax.plot_surface(x, y, z, cmap='viridis')
        ax.set_title('3D KDE of PCA Data')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('Probability Density')
        plt.show()

    # Sample 10 new points from the distribution
    new_samples = kde.resample(10).T

    # Visualize the original PCA data and the new samples
    if plot_new_samples:
        plt.figure(figsize=(8, 6))
        plt.scatter(PCA_data['PCA1'], PCA_data['PCA2'], alpha=0.5, label='Original Data')
        plt.scatter(new_samples[:,0], new_samples[:,1], color='red', label='New Samples', alpha=0.5)
        plt.title('PCA Scatter Plot with New Samples')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

    return new_samples

def reverse_map_2d_to_high_dim(
    pca_samples_2d: np.ndarray,
    pca_model: PCA,
    scaler: StandardScaler,
    features: list
) -> pd.DataFrame:
    """
    Approximate the original high-dimensional feature values for new data points 
    based on their positions in the 2D PCA space.

    This function takes an array of 2D points from the PCA space, applies the inverse
    PCA transformation, and then applies the inverse scaling to project them back into
    the original feature space. The result is a DataFrame with the approximated feature
    values for these new points, which could represent new political parties.

    Parameters:
    -----------
        pca_samples_2d (np.ndarray): A NumPy array of new samples in the 2D PCA-transformed space.
        pca_model (PCA): The PCA model that was used to reduce the dimensionality of the original data.
        scaler (StandardScaler): The StandardScaler instance used to scale the original data prior to PCA.
        features (list): A list of original feature names corresponding to the high-dimensional data.

    Returns:
    --------
        pd.DataFrame: A DataFrame containing the approximated original high-dimensional feature values.
    """

    # Apply the inverse PCA transformation
    high_dim_samples = pca_model.inverse_transform(pca_samples_2d)

    # Inverse transformation with StandardScaler
    original_feature_values = scaler.inverse_transform(high_dim_samples)

    original_features_df = pd.DataFrame(data=original_feature_values, columns=features)

    # Change column party_id from float to integer
    original_features_df['party_id'] = original_features_df['party_id'].astype(int)

    return original_features_df

def plot_valid_area_2d(
    pca_data:pd.DataFrame,
    expert_data_aggregated:pd.DataFrame,
    pca_model:PCA,
    bounds:tuple[int,int]
) -> None:
    """
    Plot an approximation of the valid area in the 2D PCA-transformed space. This is done by
    generating a large number of binary combinations that represent extreme points within the
    bounds of the original high-dimensional space, transforming these points using a fitted PCA model,
    and then constructing and plotting a convex hull of the transformed extremes to represent
    the valid area.

    Parameters:
    -----------
        pca_data (DataFrame): The PCA-transformed data to be plotted.
        expert_data_aggregated (DataFrame): The original aggregated high-dimensional data used for fitting the scaler.
        pca_model (PCA): The trained PCA model used to transform the extreme samples.
        bounds (tuple): The minimum and maximum bounds for each feature in the original space.

    Returns:
    --------
        None
    """
    fig, ax = plt.subplots()

    scaler = StandardScaler()
    scaler.fit(expert_data_aggregated)

    # Generate binary combinations
    binary_combinations = generate_binary_combinations(52, 1000000)

    # Map binary combinations to the actual bounds
    extreme_samples = bounds[0] + (bounds[1] - bounds[0]) * binary_combinations

    # Transform the extreme value samples using the fitted scaler and PCA
    scaled_extreme_samples = scaler.transform(extreme_samples)
    transformed_extremes = pca_model.transform(scaled_extreme_samples)

    # Create a convex hull of the transformed extremes
    hull = ConvexHull(transformed_extremes)

    # Plot the PCA data for comparison
    ax.scatter(pca_data['PCA1'], pca_data['PCA2'], alpha=0.5, label='PCA Data')

    # Create a polygon to represent the convex hull
    polygon_points = transformed_extremes[hull.vertices]
    polygon = Polygon(polygon_points, closed=True, alpha=0.3,color='green', edgecolor='black', label='Valid Area')

    # Add the polygon to the plot
    ax.add_patch(polygon)

    # Set labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('2D PCA Space with Valid Area from high-dimensional space')

    ax.legend()
    plt.show()

    return ax

def main():
    """
    Main function to process CHES2019 data.
    Task 1: Load and preprocess data.
    Task 2: Scale features and perform PCA.
    Task 3: Generate new samples from 2D distribution.
    Task 4: Find feature values from the high-dimensional space that map to the sampled 2D points.
    Task 5: Paint the valid area from the high-dimensional space.
    """

    # TASK 1: Data Loading and Preprocessing
    expert_file_name = 'CHES2019_experts.csv'
    party_file_name = 'CHES2019V3.csv'
    expert_raw_data, party_raw_data = load_data(expert_file_name, party_file_name)
    party_data = preprocess_party_data(party_data=party_raw_data)
    expert_data = preprocess_expert_data(expert_data=expert_raw_data, impute_data=False, remove_d=False, save_data=False)
    # Aggregate data by political party
    expert_data_aggregated = aggregate_pol_parties(expert_data=expert_data)

    # TASK 2: Scaling and PCA
    scaled_data, scaler = scale_data(expert_data_aggregated)
    combined_data = combine_data(scaled_data, party_data)
    pca_result, pca_model, features = perform_PCA(combined_data)
    plot_PCA(pca_result)

    # TASK 3: Generating new samples
    new_samples = generate_new_samples(
        PCA_df=pca_result,
        plot_3D_dist=False,
        plot_new_samples=False
    )

    # TASK 4: Find feature values from the high-dimensional space that map to the sampled 2D points
    approximated_features = reverse_map_2d_to_high_dim(new_samples, pca_model, scaler, features)

    # TASK 5: Paint the valid area from the high-dimensional space
    bounds = (0,10)
    plot_valid_area_2d(
        pca_data=pca_result,
        expert_data_aggregated=expert_data_aggregated.reset_index(),
        pca_model=pca_model,
        bounds=bounds
    )

if __name__=='__main__':
    main()