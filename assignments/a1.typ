#let title = "Assignment 01"
#let subtitle = "Learning from data and related challenges and linear models for regression"
#let due_time = "April 3 at 23:59"
#let author = "Thuvaragan S."
#let index = "210657G"
#let course_id = "EN3510"
#let course_name = "Pattern Recognition"
#let instructor = "Sampath Perera"
#let semester = "2024"
#let department = "Electronics and Telecommunication Department"
#let university = "University of Moratuwa"


#let assignment_title(title, author, course_id, professor_name, semester, due_time, body) = {
  set document(title: title, author: author)
  set page(
    paper:"a4",
    // header: locate(
    //     loc => if (
    //         counter(page).at(loc).first()==1) { none }
    //     else if (counter(page).at(loc).first()==2) { align(right,
    //           [*#author* | *#course_id: #title* | *Problem 1*]
    //         ) }
    //     else {
    //         align(right,
    //           [*#author* | *#course_id: #title* | *Problem #problem_counter.at(loc).first()*]
    //         )
    //     }
    // ),
    footer: locate(loc => if (
            counter(page).at(loc).first()==1) { none }
      else {
      let page_number = counter(page).at(loc).first()
      let total_pages = counter(page).final(loc).last()
      align(center)[Page #page_number of #total_pages]
    }))

  // block(height:25%,fill:none)
  align(center + top, image("./assets/uom.png", width:30%))
  align(center, text(17pt)[
    *#title*])
  align(center,text(13pt)[_ #subtitle _])
  align(center,text(14pt)[*#author #index*])
  align(center, text(12pt)[#datetime.today().display("[day] [month repr:long] [year]")])
  // block(height:52%,fill:none)
  
  align(center + bottom, image("./assets/entc.png", width:20%))
  align(center + bottom, text(11pt)[Submitted in partial fullfillment of the requirements for the module *#course_id* : *#course_name* from _ #department, #university _])

    // locate(loc => {
    //   let i = counter(page).at(loc).first()
    //   if i == 1 { return }
    //   set text(size: script-size)
    //   grid(
    //     columns: (6em, 1fr, 6em),
    //     if calc.even(i) [#i],
    //     align(center, upper(
    //       if calc.odd(i) { title } else { author-string }
    //     )),
    //     if calc.odd(i) { align(right)[#i] }
    //   )
    // })

//   if student_number != none {
//     align(top + left)[Student number: #student_number]
//   }

//   align(center)[= #title]
  pagebreak(weak: false)
  body
}

#show: assignment_title.with(title, author, course_id, instructor, semester, due_time)

= 1 Data pre-processing

== Feature 1 - Max-Abs Scaling
The data is sparse, with most values at or near zero there are a few large positive and negative outliers. Max-abs scaling would preserve the zero values and the relative magnitudes of the outliers, while scaling all values to a [-1, 1] range. This method maintains the sparsity of the data, which benifits certain machine learning algorithms.

== Feature 2 - Standard Scaling
The data appears to have a roughly normal distribution around zero. There's a wide range of values, but no extreme outliers compared to feature 1. Standard scaling will center the data around zero and scale it to unit variance, which is appropriate for normally distributed data to maintain its overall structure.

= 2 Learning from data

== 2.1 Data Generation
The initial data generation (in listing 1) uses random number generation for both x values and epsilon values. this means each time you run the code, you get a slightly different dataset.

== 2.2 Data Visualization

In listing 2, the `train_test_split` function uses a random state `r = np.random.randint(104)` to split the data. This random state changes with each run, resulting in different data points being assigned to the training and testing sets each time. With that the underlying data generation process also includes random noise `epsilon`, which affects the `Y` values differently in each run.

== 2.3 Linear Regression

The linear regression model is observed to be different from one instance to another because, with different training data in each iteration, the model learns slightly different parameters (slope and intercept) to best fit that particular subset of data.


#figure(
  image("./assets/a1q2fig1.png", width: 70%),
  caption: [
  10 random iterations of Linear Regression with 100 sampls
  ],
)

== 2.4 Increasing sample size

It can be observed that the linear regression models from 10000 samples are,
- More consistent
- Represent the underlying $Y=3+2X + epsilon$ better
The reason for this different behavior is that larger sample sizes provide more information about the underlying data distribution and reduce the impact of random noise and outliers. This leads to more stable and accurate models that are less sensitive to the particular subset of data used for training. This is why usually the estimates become more efficient and consistent as the sample size increases.


#figure(
  image("./assets/a1q2fig2.png", width: 70%),
  caption: [
  10 ramdom iterations of Linear Regression with 10000 sampls
  ],
)

= 3 Linear regression on real-world data

== 3.1 Loading dataset

== 3.2 Variable analysis

The dataset is intended to be used in a regression task to predict the oral temperature using the environment information as well as the thermal image readings.

+ Independent variables (features) - *33* - Consist of gender, age, ethnicity, ambient temperature, humidity, distance, and other temperature readings from the thermal images.

+ Dependent variables (targets) - *2* - `aveOralF` and `aveOralM` (oral temperature measured in fast mode and monitor mode, respectively).

== 3.3 Linear regression applicability

We *cannot directly apply* linear regression to this dataset without some preprocessing steps. Categorical variables should be addressed before applying linear regression. Convert categorical variables like Gender, Age, and Ethnicity into numerical format using techniques like one-hot encoding or label encoding. We can also consider averaging the the range of values for each category to create a numerical representation. We can also not use them in the regression model if they are not relevant to the prediction.

// Optional Enhancements:
//
// + Feature scaling: Normalize or standardize the numerical features to ensure they are on the same scale, which can improve the performance of linear regression.
//
// + Feature selection: Choose relevant features that are likely to have a strong correlation with the target variable. This step is already mentioned in the question (selecting Age and four other features).
//
// + Handle multicollinearity: Check for highly correlated independent variables and consider removing or combining them to reduce multicollinearity.
//
// + Check for linearity: Verify the assumption of linearity between independent variables and the dependent variable. If necessary, apply transformations to achieve linearity.

== 3.4 Correct approach NaN removal

The approach is not appropriate because it handles `X` and `y` separately, which can lead to misalignment of data points. It is better to remove rows with missing values from both `X` and `y` to maintain the correspondence between input features and target values.

data cleaning can be done as follows after concatnating `X` and `y` columns.

```python
# the following implementation of data cleaning is wrong
# X = X.dropna()
# y = y.dropna()

# corrected implementation
# drop rows with missing values from both X and y at the same time
data = pd.concat([X, y], axis=1)
data_cleaned = data.dropna()
X = data_cleaned[X.columns]
X.info()
y = data_cleaned[y.columns]
y.info()
```

== 3.5 Select features



== 3.6 Split data

== 3.7 Train a linear regression model
```
Intercept: 12.916980107531447
Coefficients:
    Feature  Coefficient
0       Age     0.001718
1     T_RC1     0.708483
2     T_atm    -0.051306
3  Humidity     0.001446
4  Distance     0.004664
```
== 3.8 Contribuition of features

From the selected features `T_RC1` seemed to have the most influence on the dependent feature.

== 3.9 Retrain and estimate coefficients

```
Intercept: 6.951744716410143
Coefficients:
      Feature  Coefficient
0       T_OR1    -0.068227
1   T_OR_Max1     0.637790
2  T_FHC_Max1    -0.048580
3   T_FH_Max1     0.320878
```

== 3.10 Calculate the following

```
Residual Sum of Squares: 16.109558448106856
Residual Standard Error: 0.28452162486463545
Mean Squared Error: 0.07896842376522968
R-squared: 0.7209108053457133
Standard Error of Coefficients: [
  1.37948932
  1.70083398
  1.70168529
  0.07643973
  0.08898284
]
T-values: [ 4.78065388  0.5453274  -0.21840103 -0.75100896  4.08784018]
P-values: [
  3.39193968e-06 
  5.86139053e-01 
  8.27340471e-01 
  4.53534423e-01 
  6.31488415e-05
]
```

== 3.11 Discarding features based on p-value

Yes, typically a lower p-value is significant for a statistical relationship. So we can safely ignore features with higher p-value `> 0.05`. We can consider removing features 2, 3 and 4 as they have higher p-values.


= 4 Performance evaluation of Linear regression

== 4.2 RSE calculation

#[#set align(center)

$"RSE" = sqrt("SSE"/N)$

$"RSE"_A = sqrt(9/10000) = 0.03$

$"RSE"_B = sqrt(2/10000) = 0.014$
]

Since lower RSE corresponds to better model performance, model B is better.

== 4.3 $R^2$ calculation

#[#set align(center)

$R^2 = 1 - "RSE"/"TSS"$

$R^2_A = 1 - 0.03/90 = 0.99967$

$R^2_B = 1 - 0.014/10 = 0.99860$

]

Since higher $R^2$ corresponds to better model performance, model A is better.

== 4.4 Performance metric comparision

$R^2$ is typically fair when comparing 2 models because it is independent of the scale of the data while also accounting for the inherent variablility in the dataset. This is not the case with RSE, which will have higher value errors for larger datasets. Even though model A and model B have the same sample size in this case, $R^2$ will still be better and fair campared to RSE as it accounts for the inherent variability in the dataset.

#pagebreak()

= 5 Linear regression impact on outliers

== 5.1 Modified loss functions

#[#set align(center)
$
L_1(bold(w)) = 1/N sum_(i=1)^N (r_i^2/(a^2+r_i^2)) = 1/N sum_(i=1)^N (L_(1,i))
\
L_2(bold(w)) = 1/N sum_(i=1)^N (1 - e^(-2 abs(r_i)/a)) = 1/N sum_(i=1)^N (L_(2,i))
$
]

== 5.2 Analysis w.r.t. $a arrow 0$

#[#set align(center)
$
// L_1(bold(w)) = 1/N sum_(i=1)^N (r_i^2/(a^2+r_i^2)) = 1/N sum_(i=1)^N (L_(1,i))
// \
"as" a arrow 0 space : space
L_1(bold(w)) approx 1/N sum_(i=1)^N (r_i^2/r_i^2) = 1/N sum_(i=1)^N 1 = 1
$
]

#[#set align(center)
$
// L_2(bold(w)) = 1/N sum_(i=1)^N (1 - e^(-2 abs(r_i)/a)) = 1/N sum_(i=1)^N (L_(2,i))
// \
"as" a arrow 0 space : space
L_2(bold(w)) approx 1/N sum_(i=1)^N (1 - e^(-infinity)) approx 1/N sum_(i=1)^N (1-0) = 1
$
]

Considering the situation where residuals are relatively larger than the hyper-parameter `a` or `a` being relatively small or close to zero, both loss functions reach 1. This behaviour is contrastive of usual loss functions which are less robust in the presence of outliers. Hence it can be observed that the objective of reducing the impact of outliers is achieved through clamping the loss value as residual values increase. Compared to a standard loss function such as `MSE` which doesn't limit the effect of outliers at all, this will reduce the impact of outliers in the dataset.

== 5.3 Choosing appropriate loss function

Analysing the loss functions reveals that both functions are good at handling residuals. $L_2(w)$ can be choosen on the basis of aggressive clamping of residual values using exponential scaling.

A relatively lower value of `a` will suffice for $L_2(w)$ in the range of 5 to 20 to have a balance between clamping and not restricting data too much.

#figure(
  image("./assets/a1q5fig1.png"),
)

#figure(
  image("./assets/a1q5fig2.png"),
)
