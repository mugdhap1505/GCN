# Almetrics Bot Analysis and Prediction using Network Science and Graph Neural Networks

## Introduction
An internet bot is an automated software application which can run any range of tasks respectively.
Social bots interact with users with a purpose to generate content that promotes a particular viewpoint.
Bot accounts are problematic because they can manipulate information, and promote unverified information, which can adversely affect public opinion on
various topics which can affect the decision making and ranking systems. 
 <p align="center">
    <img src="/Images/Picture1.png" alt="Image" width="400" height="400" />
</p>

## Objective 
1. To find the existence of automated agents in the academic related twitter accounts.
2. To analyze the behavior of automated agents and their classification over scientific documents of Twitter.
3. Distinguish between the automated accounts and humans.
4. Lastly, map the altmetrcs data into a network and perform network analysis.

## Altmetrics 
It is a qualitative metric that evaluates the importance of the scientific documents concerning their usage and discussion on the web.

### Benifits: 
1. It captures elements of social impact.
2. Complements traditional metrics.
3. It offers speed and discoverability.

## Description of Data:
1. Bots Annoted dataset.
2. Altmetrics dataset.

## Project Breakdown:
1. Network Creation.
2. Network Visualization.
3. Network Analysis.
4. Graph Convolution Network.

### Network creation:
1. The annoted bots dataset and the altmetrics dataset has a common field; altmetrics-id.
2. Using the almetrics-id, we queried both datasets and chose the map records.
3. We created a network where each node represents a tweet and an edge exists between two tweets if they tweeted or retweeted.
4. The final network consists of 16210 nodes and 31270 edges.

### Network Visualization:

Tool used : Gephi 

 <p align="center">
    <img src="/Images/Picture2.png" alt="Image" width="800" height="400" />
</p>


### Network Analysis:
Information about the relative importance of nodes and edges in a graph can be
obtained through centrality measures.

### Centrality: 
Centrality is the measure of importance of a node in a network.

### Centrality Measures:
1. Degree Centrality.
2. EigenVector Centrality.
3. Closeness Centrality.
4. Betweeness Centrality.

## Degree Centrality:
1. Degree centrality is a simple centrality measure that counts how many neighbors a node has.
2. Degree of a node is the sum of incoming and outgoing edges of a node.
3. It is one of the simplest centrality measures.

<p align="center">
    <img src="/Images/Picture3.png" alt="Image" width="600" height="400" />
</p> 


<p align="center">
    <img src="/Images/Picture4.png" alt="Image" width="600" height="400" />
</p> 

## EigenVector Centrality:
1. Eigenvector centrality measures a node’s importance while considering the importance of its neighbors.
2. For example, a node with 300 relatively unpopular friends on Facebook would have lower eigenvector centrality than someone with 300 very popular friends.
3. It is sometimes used to measure a node’s influence in the network.

## Findings:
1. The Degree distribution and Eigenvector centrality nearly follows power law distribution.
2. This implies that there are few nodes that act like hubs in the network while most of the nodes follow them.
3. It is highly known that most of the online social networks follow power law. Our network also follows the same. There are 13 major communities of the network.
4. We found that each community contains bots, which are actively involved in disseminating information across the network.

## Graph Convolutional Networks (GCNs):
1. We observed that bots exist in each community and spread across the network.
2. This observation gives up hope to train a GCN model that distinguish between human and bot.
3. Since GCN learns the local and global positions of nodes. We can adopt a semi-supervised learning model to train on the labeled data.
4. GCNs involves two main steps the aggregation and the model update
![Before After Previews of 12.SPE](/Images/Picture5.png?raw=true "Network visualization") 
5. The first equation is the aggregation step that aggregates feature vectors from all the neighbor nodes. The second equation is the model update step.

 <p align="center">
    <img src="/Images/Picture6.png" alt="Image" width="600" height="400" />
</p>

## Experimental setup and results:
1. Learning rate is set to 0.01 and the number of epochs are set to 50.
2. The train test split is set to 70:30.
3. The training accuracy results to 72.5%
4. Testing accuracy accounts to 68%

<p align="center">
    <img src="/Images/Picture7.png" alt="Image" width="600" height="400" />
</p>
