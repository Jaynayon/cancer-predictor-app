import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.stylable_container import stylable_container
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def color(color_name: str) -> str:
    """Converts a color name to its corresponding CSS color code."""
    color_map = {
        "light-blue-70": "#00c0f2",
        "orange-70": "#ffa421",
        "blue-green-70": "#00d4b1",
        "blue-70": "#1c83e1",
        "violet-70": "#803df5",
        "red-70": "#ff4b4b",
        "green-70": "#21c354",
        "yellow-80": "#faca2b",
    }
    return color_map.get(color_name, "#000000")  # Default to black if color_name is not found

# Replicate streamlit_extras colored header: add color header feature.
def colored_header(
    label: str = "Nice title",
    description: str = "Cool description",
    color_name: str = "red-70",
    header_color: str = "#31333f"
):
    """
    Shows a header with a colored underline and an optional description.

    Args:
        label (str, optional): Header label. Defaults to "Nice title".
        description (str, optional): Description shown under the header. Defaults to "Cool description".
        color_name (str, optional): Color of the underline. Defaults to "red-70".
    """
    if color_name is None:
        color_name = next(HEADER_COLOR_CYCLE)  # Ensure HEADER_COLOR_CYCLE is defined elsewhere

    st.write(
        f'<h3 style="color: {header_color}; margin-top: 0; margin-bottom: 0;">{label}</h2>',
        unsafe_allow_html=True,
    )
    st.write(
        f'<hr style="background-color: {color(color_name)}; margin-top: 0;'
        ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">',
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)

# Show app title and description.
st.set_page_config(page_title="Cancer Prediction", page_icon="🎯")
st.title("🎯 Cancer Prediction")
st.write(
    """
    This app provides significant insights into cancer using a dataset 
    containing information on 1,000 cancer patients. Users can explore 
    various fields and discover correlations between them.
    """
)

# Custom CSS for st.info-like box with a different color
st.markdown(
    """
    <style>
    .custom-info {
        background-color: rgba(255, 255, 255, 0.12); 
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #6b6b6b;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

with stylable_container(
        key="container_black",
        css_styles="""
            {
                background-color: #000;
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                color: #ffffff;
            }
            """,
    ): 
    with st.container():
        # Add a section for introduction
        colored_header(
            label="Introduction 🤓",
            description="",
            color_name="orange-70",
            header_color="white"
        )
        st.write(
            """
            The dataset used for this analysis is sourced from Kaggle, titled 
            :blue[**Cancer Patients Data**]. It contains 1,000 records of cancer patients 
            with 24 variables related to demographic information, environmental factors, 
            and health symptoms. The goal of this analysis is to gain insights into 
            the central tendencies, spread, and patterns within the data, with a focus 
            on understanding risk levels and other key factors.
            """
        )

        st.markdown(
            """
            <div class="custom-info">
                The dataset originally included a <strong style="color:red;">Patient ID</strong> field, 
                but we removed it as it doesn't contribute to the analysis or any meaningful insights.
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="custom-info">
                All these fields accept values from 1 to 9, except for <i>Gender</i>, which accepts values <strong>1</strong> and 
                <strong>2</strong>, and  <strong style="color:yellow;">Level</strong> column, which represents the severity 
                levels (<i>Low, Medium, High</i>), has been label encoded for analysis as (1, 2, 0), respectively.
            </div>
            """, unsafe_allow_html=True
        )


        # Load the dataset and test if it is loaded correctly
        try:
            data = pd.read_excel("./dataset/cancer patient data sets.xlsx")
            data = data.drop('Patient Id',axis=1)

            # Encode and transform 'level' field
            le = LabelEncoder()
            encoded_level = le.fit_transform(data['Level'])
            encoded_level_df = pd.DataFrame(encoded_level, columns=['Level'])

            # Replace old 'level' field with new one
            data = data.drop('Level',axis=1)
            data=pd.concat([data,encoded_level_df],axis=1)
            
            # Display the first few rows of the data
            st.write(data.head())
        except FileNotFoundError:
            st.error("The dataset file was not found. Please check the file path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with stylable_container(
        key="container_default",
        css_styles="""
            {
                background-color: #f2f2f2;
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                color: #000;
            }
            """,
    ): 
    with st.container():
        # Add a section for introduction
        colored_header(
            label="Key Statistics",
            description="",
            color_name="blue-green-70",
            header_color="black"
        )
        st.write(
            """
            Descriptive statistics were computed for each of the numerical variables in the dataset. 
            These include the _mean, median, standard deviation, and ranges_. The table below summarizes 
            the key descriptive statistics for some of the key features in the dataset:
            """
        )

        # Load the dataset and test if it is loaded correctly
        try:
            # Dataset fields
            categories = [
                'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 
                'Occupational Hazards', 'Genetic Risk', 'Chronic Lung Disease', 
                'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 
                'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 
                'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 
                'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'
            ]

            mean_values = data.mean().values
            median_values = data.median().values
            mode_values = data.mode().values[0]  # Get mode as an array if applicable
            std_dev_values = data.std().values
            variance_values = data.var().values
            min_values = data.min().values
            max_values = data.max().values
            range_values = max_values - min_values
            percentiles = data.quantile([0.25, 0.50, 0.75])
            percentile_25_values = percentiles.loc[0.25].values
            percentile_50_values = percentiles.loc[0.50].values
            percentile_75_values = percentiles.loc[0.75].values

            # Create a DataFrame to represent the statistics table
            statistics_df = pd.DataFrame({
                'Category': categories,
                'Mean': mean_values,
                'Median': median_values,
                'Mode': mode_values, 
                'Standard Deviation': std_dev_values,
                'Variance': variance_values,
                'Min': min_values,
                'Max': max_values,
                'Range': range_values,
                '25th Percentile': percentile_25_values,
                '50th Percentile': percentile_50_values,  # Same as median
                '75th Percentile': percentile_75_values
            })

            
            # Display the first few rows of the data
            st.write(statistics_df.style)
        except FileNotFoundError:
            st.error("The dataset file was not found. Please check the file path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        st.write(
            """
            Notable values include the average age, gender, and level. The rest of the fields had an 
            average of **3-4**, while the average number of male patients was higher than that of female 
            patients, with a mean of **1.4**. Lastly, the average level of cancer severity (with values 
            ranging from 0 to 2) had a mean of **0.9**.
            """
        )

with stylable_container(
        key="container_black",
        css_styles="""
            {
                background-color: #000;
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                color: #ffffff;
            }
            """,
    ): 
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            data['Age'].hist(bins=10)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Frequency')

            st.pyplot(plt)
            plt.clf() 

        with col2:
            # Create the box plot to show Age distribution by Gender
            sns.boxplot(x='Gender', y='Age', data=data)

            # Add title and labels
            plt.title('Age Distribution by Gender')
            plt.xlabel('Gender (Male = 1, Female = 2)')
            plt.ylabel('Age')

            st.pyplot(plt)
            plt.clf() 

        st.write(
            """
            The average age of patients in the dataset ranges from the late :blue[**30s**] to early 
            :blue[**40s**], with the youngest being 14 and the oldest 73. Interestingly, by gender, 
            the average age for males is in the early 30s to late 40s, while for females, 
            it ranges from the mid-20s to late 30s. One could infer that females may be more 
            susceptible to cancer at an earlier age than males.
            """
        )

with stylable_container(
            key="container_default",
            css_styles="""
                {
                    background-color: #f2f2f2;
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    color: #000;
                }
                """,
        ): 
        with st.container():
            col1,col2 = st.columns([1.120,0.880])

            with col1:
                data['Gender'].value_counts().plot(kind='bar')
                plt.title('Gender Distribution')
                plt.xlabel('Gender (1=Male, 2=Female)')
                plt.ylabel('Count')
                st.pyplot(plt)
                plt.clf() 
                

            with col2:
                data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title('Gender Proportion')
                
                st.pyplot(plt)
                plt.clf() 

            st.write(
                """
                Based on a thousand entries, males make up around **60%** (59.8%) of the dataset, 
                while females account for **40%** (40.2%). The ratio between males and females is 
                approximately 1:0.66. It could be the case that although females may be more 
                susceptible to cancer at an earlier age, they may not acquire it as frequently.
                """
            )

with stylable_container(
        key="container_black",
        css_styles="""
            {
                background-color: #000;
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                color: #ffffff;
            }
            """,
    ): 
    with st.container():
        col1, col2, col3 = st.columns([0.5,2,0.5])

        with col2:
            # Assuming 'Gender' is already mapped to Male and Female and 'Level' to low, medium, high
            # Create the bar plot
            sns.countplot(x='Gender', hue='Level', data=data)

            # Add title and labels
            plt.title('Cancer Levels by Gender')
            plt.xlabel('Gender (Male = 1, Female = 2)')
            plt.ylabel('Count of Patients')

            # Show the plot
            st.pyplot(plt)
            plt.clf()
        
        st.write(
            f"""
            Cancer severity levels—high, low, and medium—are represented by values 0, 1, 2, respectively. 
            Compared to females, males have the :red[_highest_] count for high severity levels, with around **<u>250</u>** males 
            and about **<u>110</u>** females. However, for low severity, females are almost :red[_on par_] with males, both nearing 
            the **<u>150</u>**-count mark, with females slightly exceeding it. Lastly, nearly **<u>200</u>** males have medium severity, 
            while females are around the **<u>140</u>**-count mark.
            """, unsafe_allow_html=True
        )
        st.write(
            """
            Notable insights include males ranking :red[**first**] for the highest number of high-severity cancer cases. 
            Interestingly, despite a _<u>20%</u>_ population difference, females :red[**match or surpass**] males in low severity 
            cancer cases. Lastly, while males :red[**lead**] in medium severity cancer cases, females come close, with 
            around _<u>two-thirds</u>_ the number of cases as males.
            """, unsafe_allow_html=True
        )

with stylable_container(
            key="container_default",
            css_styles="""
                {
                    background-color: #f2f2f2;
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    color: #000;
                }
                """,
        ): 
        with st.container():
            # Get the list of column names (fields) for user selection
            columns = data.columns.tolist()

            col1,col2 = st.columns(2)

            # Create two dropdowns for selecting the fields
            with col1:
                field1 = st.selectbox('Select first field:', columns)
            with col2:
                field2 = st.selectbox('Select second field:', columns)

            # Display the selected pairplot only when the user has selected both fields
            if field1 and field2:
                # Plot a scatter plot between the selected fields
                st.write(f"{field1} vs {field2}")
                
                 # Scatter plot
                fig1, ax1 = plt.subplots()
                data.plot(kind='scatter', x=field1, y=field2, ax=ax1)
                ax1.set_title(f'{field1} vs {field2}')
                st.pyplot(fig1)
                plt.clf()

                try:
                    # Pairplot (Seaborn)
                    sns.pairplot(data[[field1, field2]])
                    
                    # Convert the seaborn pairplot into a Matplotlib figure to pass to st.pyplot()
                    st.pyplot(plt) 
                except Exception as e:
                    st.markdown(
                        """
                        <div class="custom-info" style="color:black">
                        Try selecting a different field for the pair plot to explore new relationships 
                        between variables. This will allow you to visualize how two different fields 
                        interact and identify potential correlations or patterns in the data.
                        </div>
                        """, unsafe_allow_html=True
                    )
                    # st.info(
                    #     """
                    #     Try selecting a different field for the pair plot to explore new relationships 
                    #     between variables. This will allow you to visualize how two different fields 
                    #     interact and identify potential correlations or patterns in the data.
                    #     """,
                    #     icon="✍️",
                    # )