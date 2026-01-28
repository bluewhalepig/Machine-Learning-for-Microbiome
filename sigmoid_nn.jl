
# Arranging the input data we need for model: 
	# Access libraries: CSV 
	using CSV 
    using DataFrames
	using Random 

	# Load the data 
	data = CSV.read("ECHO_data.csv", DataFrame)
    
    # Select abundance data columns 
    abundance_data = select(data, Not(:study_name, :datagroup, :site, :datacolor, :datasource, :visit,
            :westernized_cat, :subject_id, :ageMonths, :sample))
    
    # Select metadata columns 
    meta_data = select(data, [:westernized_cat, :subject_id, :ageMonths])
    
	# Select X (input) and y(output) of the model 
	X = abundance_data # multivariate input: abundance data 
    y = meta_data.ageMonths # univariate output: age in months 

	# scale y for sigmoid outputs
	y_min = minimum(y)
	y_max = maximum(y)
	y_scaled = (y .- y_min) ./ (y_max - y_min)
	
# Compoments we need to build RFR model:
	# 1. Access libraries: MLJ, DecisionTree 

	using MLJ
	using MLJFlux
	using Flux
	using StatsBase
	using Optimisers

    #import Pkg
    #Pkg.add(["MLJ", "MLJDecisionTreeInterface", "DecisionTree", "StatsBase"])
   
    # 2. Load the model of decision tree that you want
	NeuralNetworkRegressor = MLJ.@load NeuralNetworkRegressor pkg=MLJFlux
	# Instantiate a nn regressor and use to build machine with input (X) and output (y)
	
		## Parameters of NN 
			# builder = defines structure of nn 
			# loss = loss function used to measure how well the model performs 
				# means square root error: Flux.mse (often used for regression)
			# optimiser = optimization algorithm used to minimize the loss function
				# controls how model weights are updated after each epoch 
				# Adam optimizer(with learning rate of 0.001): Flux.Adam(0.001) 
			# epochs = number of training iterations 
			# batch_size = number of samples that will be used in each batch of training
			# rng = random number generator seed for initializing weights 
			# validation = can define a separate validation dataset - tunes hyperparameters! different from test set(default is to use portion of training set)
			# early_stopping = stops training early if the validation loss doesn't improve (default is none)
		#############
	
	# Define the input dimension 
	input_dim = size(X,2) # number of columns (150 features)
	# Define the number of nodes in hidden layers
	hidden1 = 16 
	hidden2 = 8 

	# Set up neural network regressor model
	nn_model = NeuralNetworkRegressor(

		builder = MLJFlux.MLP(hidden=(16, 8), σ=sigmoid),  # multilayer perceptron 2 hidden layers
		loss = Flux.mse, #means square error
		optimiser = Optimisers.Adam(0.001), #Adam: adaptive gradient descent
					# controls how weights are updated, 0.001 is learning rate
		epochs = 10,
					# num of times the model trains on data
		batch_size = 20,

		# default stuffff 
		rng = TaskLocalRNG(), #random numner generator 
		lambda = 0,
		alpha = 0, # weight decay to penalize large weights 
		optimiser_changes_trigger_retraining = false, 
		acceleration = CPU1{Nothing}(nothing), 
		embedding_dims = Dict{Symbol, Real}()
	)



	# Build machine using nn model 
    mach = machine(nn_model, X, y_scaled) 

    # 3. Partition the indexes of the rows into train and test groups 
    train, test = partition(eachindex(y_scaled), 0.7) #hold out validation with test 


	# 4. Train the data at train indexes 
	fit!(mach, rows = train)

######################
# Predictions
######################
	# 5. Test the model using text indexes of the abundance data 
	yhat_test = MLJ.predict(mach, X[test, :]) #outputs are predicted ages (yhat) 
	yhat_train = MLJ.predict(mach, X[train, :])

	# Inverse scaling
	inverse_scale(v) = v .* (y_max - y_min) .+ y_min
	yhat_test  = inverse_scale(yhat_test)
	yhat_train = inverse_scale(yhat_train)

	# 6. Calculate the mean abs error for training set and test set
	mae_test = mean(abs.(yhat_test - y[test])) 
	mae_train = mean(abs.(yhat_train-y[train]))

######################
# Data visualizations 
######################
using CairoMakie

#Scatterplot of ground and truth --> visualize model adequacy 

# Create figure 
fig = Makie.Figure()

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



# Calculate correlation between truth and yhat in training and test set 
corr_test = cor(y[test], yhat_test)
corr_train = cor(y[train], yhat_train)


# Scatter plot of training and test samples yhat 
save("models/sigmoid_nn/scatter_sigmoid_only.png", fig) 

### END 