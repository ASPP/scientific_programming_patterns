# Before the start

- Ask: who has written a class before? should be paired with someone who has not

# Introduction


End of intro
- We’ll show some code, and put it together and break it apart several times to illustrate how to make it practical to use, but flexible 
Then we’ll discuss how to do a similar thing at a research project level

# Chapter 1, classes

Before starting with the notebook
- We’ll start by looking at putting things together that belong together
- Key question: How do we know what does?

- We know data structures that groups data together, lists, dictionaries, etc; and modules and packages that organize code with a similar scope
- Here we’ll look at functions and classes as a way to group data and functionality together in a way that is easy to use but remain flexible to changes



Walker saga

- Chapter 1: put together

- create a class (exercise)

  - before the exercise
	- What are the smells?
		Examples:
		# current_i, current_j, sigma_i, sigma_j needed all over the place
		# not efficient: mgrid calculated every time
		# defining the parameters of a run is tedious: define all the single variables


  - after the exercise

	- Classes group data and functions that belong together
		- Classes are used to put together sets of parameters that belong together and would be otherwise passed over and over to a set of functions
		- Classes are templates for bundles of data and “methods”, i.e. functions that have access to the data stored in an instance

	- How is the code with the class better?
		- all parameters are specified once at the start, grouped together nicely		- the signature of most functions is simplified

	- What belongs and do not belong to a class? What have you put into it?
		- Anything for which you can imagine to write 5 different variants depending on your mood
		- For example, plotting code: write functions that take instances and create a plot instead. This also helps separating the dependencies from plotting libraries from the computational code. Related to Model-View pattern
		- Smell: a method that does not use any of the data in the instance

	- Which methods should be private?
		- think about the “interface”, i.e. the part of the class that an external user is supposed to use directly

	- Design questions
		-  should i, j belong to the class?		-  should random_state belong to the class?


- Chapter 2: break out

  - on the slide "the smells of the Walker constructor"

	- Look at Walker class, if/elif part – that’s an example of having multiple ways of initializing the activation_map. 
		- What do you think of it? 
		- Have you seen this kind of situation before?
	- What happens if we want to add a new activation map type? 
	- What if a colleague wants to do the same?
	- What happens if I want to test that one of the initializations is correct?
	- What do you think is the smell, and how would you fix it?


  - Live coding: add a factory method for the context map

	- after live coding
		- What have we accomplished?
		- the activation map initialization varies independently from the class
		- we can now add a new activation map initialization without changing the code -> more flexibility, extendability

  - serialization
	- Open discussion: what happens when the class changes?
	- Discussion in to_json: what happens if the code breaks before the array is saved? What to do about it?

  - Live coding Breaking out 

    - after the live coding

	- By breaking out the parts that can vary (initializers), the model code is much simplified. All you need to do is define the interface for the initialization method
	- You can add a new initialization method without touching the model! Changes to the initializers also do not affect the Walker. The code became much more flexible
	- We can test the initializer methods much more easily; the model as well, as we can now define a new TestInitializer that sets the ”x” to a convenient initial value
	- This pattern is called “dependency injection”
	- what to do if one initializer had a parameter?


  - break out the next step probability (exercise)

	- ... let's say we want to have a rectangular, uniform next step distribution
	- talk about the fact that we can pass anything as long as you can call it as initializer(size)

    - after the exercise

	- The Walker class has less parameters!
	- its responsibilities are clearer
	- What about serialization? now serialization is going to be harder


- Chapter 3: project level

  - after the trinity

	- Has it ever happened to you that you had a figure and no idea which version of the code produced it?
	- How do you typically manage your research code?
	- Is there one folder with everything? How is it structured? Does each run of the code overwrite the previous files? Do you have dates in the filenames?
	- Which metadata do you save?
	- Where does the data live?
	- Reproducibility: what about randomness (sampling?)

  - after showing the project organization

	- Which of these things should be in git? Which should not?
	- Should „research_folder“ be git versioned? Or separate git repos for each subfolder?
	- Which kind of thing is happy in a notebook? What should be a .py script? A package?
	- How do we keep track of versions? How do we write the „meta“ file?
	- Python git package can read the hash
	- Dated folders
	- Pip freeze

	- additional summary stuff

	  - break out run parameters and code
	    - what is parameter injection
	    - manager code: small amount of code reads the run parameters, create the appropriate classes, injects the parameters, and runs
	    - it’s a coordination job, all the science is done elsewhere
	  - separate calculation and analysis of results
	  - recurrent research project concepts: projects, experiments, runs
	  - break out data and code
	    - data not in git
	    - what is lineage, versioning
	    - Some “input” data is common to the whole project






Saved snippet:
```
def evaluate_position(current_i, current_j, sigma_i, sigma_j, activation_map):
    size = activation_map.shape[0]
    next_step_map = next_step_probability(current_i, current_j, sigma_i, sigma_j, size)
    selection_map = compute_selection_map(next_step_map, activation_map)
    return selection_map[current_i, current_j]
```
