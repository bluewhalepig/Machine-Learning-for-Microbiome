
# Arranging the input data we need for model: 
	# Access libraries: CSV 
	using CSV 
    using DataFrames

	# Load the data 
	data = CSV.read("ECHO_data.csv", DataFrame)
    
    # Select abundance data columns 
    abundance_data = select(data, Not(:study_name, :datagroup, :site, :datacolor, :datasource, :visit,
            :westernized_cat, :subject_id, :ageMonths, :sample))
    
    # Select metadata columns 
	meta_data = select(data, [:westernized_cat, :subject_id, :ageMonths])
    
	
# Compoments we need to build RFR model:
	# 1. Access libraries: MLJ, DecisionTree 
	using MLJ
	using MLJDecisionTreeInterface
	using DecisionTree
	using StatsBase

    #import Pkg
    #Pkg.add(["MLJ", "MLJDecisionTreeInterface", "DecisionTree", "StatsBase"])
   
    # 2. Load the model of decision tree that you want
    RandomForestRegressor = MLJ.@load RandomForestRegressor pkg=DecisionTree

    # Instantiate a RandomForest Regressor and use to build machine with input (X) and output (y)
    reg = RandomForestRegressor()
    X = abundance_data # multivariate input: abundance data 
    y = meta_data.ageMonths # univariate output: age in months 
    mach = machine(reg, X, y) 

    # 3. Partition the indexes of the rows into train and test groups 
    train, test = partition(eachindex(y), 0.7)


	# 4. Train the data at train indexes 
	MLJ.fit!(mach, rows = train)

	# 5. Test the model using text indexes of the abundance data 
	yhat_test = MLJ.predict(mach, X[test, :]) #outputs are predicted ages (yhat) 
	yhat_train = MLJ.predict(mach, X[train, :])

	# 6. Calculate the mean abs error for training set and test set
	mae_test = mean(abs.(yhat_test - y[test])) 
	mae_train = mean(abs.(yhat_train-y[train]))


######################
# Data visualizations 
######################
using CairoMakie

#Scatterplot of ground and truth --> visualize model adequacy 

# Create figure 
fig = Figure()

# Create axis with titles and labels
ax = Axis(fig[1,1],
	title = "Scatter plot of Training and Test data for age prediction",
	xlabel = "ground truth age in months",
	ylabel = "prediction age in months"
)

# Plot scatterplot on axis 
sc = scatter!(ax, y[train], yhat_train, color=:blue)
scatter!(ax, y[test], yhat_test, color=:red)

# Place legend on figure 
Legend(fig[1,2], 
	[MarkerElement(color = :blue, marker = :circle), MarkerElement(color = :red, marker = :circle)], 
	["Train", "Test"]) # position of legend, markers, label
#save("scatter.png", fig)


# Calculate correlation between truth and yhat in training and test set 
corr_test = cor(y[test], yhat_test)
corr_train = cor(y[train], yhat_train)


# Scatter plot of training and test samples yhat 

save("scatterplot_yhat.png", fig)

### END 