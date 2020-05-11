# YahtzeeLearning
An algorithmic attempt to make the game of Yahtzee more interesting

How does one characterize the depth of a game? The issue can be addressed through the lens of steady progress: as an agent learns the ropes, he, she, or it would likely find it interesting to steadily improve. In other words, a good game is arguably one that offers a path to becoming more skilled in a *more or less linear fashion*. This desirable property informs the intuition for a strategy ladder, an idea originally proposed by [Lantz et al](http://julian.togelius.com/Lantz2017Depth.pdf). In this work, the relationship between available resources and strategy strength is explored. Plotting the former against the best possible score for the latter, across all possible strategies at that level of resources, should yield a curve that informs how "deep" a game is.

Here, I attempt to probe the shape of Yahtzee's strategy ladder via reinforcement learning. Can varying the upper bonus threshold and/or value bring about a more gradual curve?

Results are currently noisy, but with a bit more compute power a clearer relationship should emerge.

![Preliminary results](all_ladders_newer.png)

Contents:

* yahtzee.py
	* Program that determines how well a neural network can learn to play variant y of Yahtzee from a limited set of training examples generated for y.

* query_optimal_fast.py
 	* Stand-alone program that computes the optimal strategy for a given variant of Yahtzee

* query_optimal.py
	* Slower, but more organized implementation of query_optimal_fast

Usage: run [program] -help for a list of command-line arguments