import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import math
import time
from typing import List, Tuple, Union
import math as math 
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

def compute_haversine(lat_long_tuple_1: tuple, lat_long_tuple_2: tuple) -> float:
    lat1, lon1 = lat_long_tuple_1
    lat2, lon2 = lat_long_tuple_2
    R_km = 6371.0088  # mean Earth radius in km
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_km * c

def haversine_list_computer(initial_tuple, list):

    haversine_distance_list = []

    for i, entry in enumerate(list):
        haversine_distance_list.append(compute_haversine(initial_tuple, entry))

    return haversine_distance_list

def get_city_coordinates(city: str, state: str, country: str,
                         fallback: str = "nan") -> Tuple[float, float]:
    """
    Returns (latitude, longitude) of a given city, state, and country.
    
    If not found, returns:
        - (math.nan, math.nan) if fallback="nan"
        - (0.0, 0.0) if fallback="zero"
    """
    geolocator = Nominatim(user_agent="city-locator", timeout=5)

    query = f"{city}, {state}, {country}"
    location = geolocator.geocode(query)
    
    if location is None:
        if fallback == "nan":
            return (math.nan, math.nan)
        elif fallback == "zero":
            return (0.0, 0.0)
        else:
            raise ValueError("Invalid fallback option. Use 'nan' or 'zero'.")
    
    return (location.latitude, location.longitude)

def batch_get_coordinates(locations: List[Tuple[str, str, str]],
                          fallback: str = "nan",
                          pause: float = 0.05) -> List[Tuple[float, float]]:
    """
    Batch geocode a list of (city, state, country) tuples.
    Returns a list of (lat, lon) tuples.
    
    Parameters:
        locations : list of (city, state, country)
        fallback  : "nan" or "zero" for missing results
        pause     : seconds to wait between queries (default: 1.0, for Nominatim etiquette)
    """
    coords = []
    for city, state, country in locations:
        latlon = get_city_coordinates(city, state, country, fallback=fallback)
        print(len(coords), latlon)
        coords.append(latlon)
        time.sleep(pause)  # be nice to the server
    return coords

def bfs(G, source):
    """ return a dictionary that maps node-->distance for all nodes reachable
        from the source node, in the unweighted undirected graph G """
    # set of nodes left to visit
    nodes = deque()
    nodes.append(source)
    
    # dictionary that gives True or False for each node
    visited = {node:False for node in G}
    visited[source] = True
    
    # Initial distances to source are: 0 for source itself, infinity otherwise
    dist = {node: np.inf for node in G}
    dist[source] = 0
    
    # while (container) is shorthand for "while this container is not empty"
    while nodes:
        # take the earliest-added element to the deque (why do we do this instead of popright?)
        node = nodes.popleft()
        
        # visit all neighbors unless they've been visited, record their distances
        for nbr in G.neighbors(node):
            if not visited[nbr]:
                dist[nbr] = dist[node] + 1
                visited[nbr] = True
                nodes.append(nbr)
    return dist

def components(G):
    """ return a list of tuples, where each tuple is the nodes in a component of G """
    components = []
    
    nodes_left = set(G.nodes())
    while nodes_left:
        src = nodes_left.pop()
        dist = bfs(G, src)
        component = [node for node in dist.keys() if dist[node] < np.inf]
        components.append(component)
        nodes_left = nodes_left - set(component)
    return components

def time_obtain_list(list, clock_time_given):
    """given a list where the ith elemnt is a "MM/DD/YYYY" or "MM/DD/YYYY HOUR:MINUTE" string, 
    return a list where the ith element is that same string converted to total minutes"""

    dmy_occurence_list = []
    clock_occurence_time_list = []

    for i, entry in enumerate(list):
        dmy_occurence_list.append(entry.split(' ')[0])
        if clock_time_given == True:
            clock_occurence_time_list.append(entry.split(' ')[1])

    minute_occurence_array = [] # in minutes

    if clock_time_given == True:
        for i, clock in enumerate(clock_occurence_time_list):
            hour = int(clock.split(':')[0])
            minute = int(clock.split(':')[1])
            minute_occurence_array.append(hour*60 + minute)

    minute_occurence_array = np.asarray(minute_occurence_array)

    minute_occurence_array_1 = []

    for i, entry in enumerate(dmy_occurence_list):
        month, day, year = [int(x) for x in entry.split('/')]
        year_to_minutes = year * 525600
        month_to_minutes = month * 43800
        day_to_minutes = day * 1440
        minute_occurence_array_1.append(year_to_minutes + month_to_minutes + day_to_minutes)    

    minute_occurence_array_1 = np.asarray(minute_occurence_array_1)

    if clock_time_given == True:
        total_minute_occurence_list = minute_occurence_array_1 + minute_occurence_array
    else:
        total_minute_occurence_list = minute_occurence_array_1

    return total_minute_occurence_list.tolist()

def data_acquisition(file_name):
    file_name = file_name
    df = pd.read_csv(file_name, dtype=str, skiprows = 0, header=None)
    # df = pd.read_csv(file_name, dtype={'a': np.float64, 'b': np.int32, 'c': 'Int64'})
    data = df.to_numpy()
    date_occurence_array = data.T[1]
    city_array = data.T[2]
    state_array = data.T[3]
    country_array = data.T[4]
    ufo_shape_array = data.T[5]
    ufo_description_array = data.T[6]
    date_reported_array = data.T[7]

    # USA MASK
    usa_mask = (country_array == 'USA')
    country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array = [arr[usa_mask] for arr in [country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array]]

    # lat_long_list = [get_city_coordinates(city_array[i], state_array[i], country_array[i]) for i in range(len(city_array))]
    # # lat_long_list = batch_get_coordinates(csc_list)
    # np.savetxt('latitude_longitude_data.csv', lat_long_list, delimiter=',')

    latitude_list, longitude_list = np.genfromtxt("latitude_longitude_data.csv", delimiter=',', unpack=True)

    nan_mask = ~np.isnan(latitude_list)
    latitude_list, longitude_list, country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array = [arr[nan_mask] for arr in [latitude_list, longitude_list, country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array]]


    longitude_mask = (longitude_list < -67)
    latitude_list, longitude_list, country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array = [arr[longitude_mask] for arr in [latitude_list, longitude_list, country_array, city_array, state_array, ufo_shape_array, date_reported_array, date_occurence_array, ufo_description_array]]

    csc_list = [(city_array[i], state_array[i], country_array[i]) for i in range(len(city_array))]

    lat_long_tuple_list = []
    for i in range(len(latitude_list)):
        lat_long_tuple_list.append((float(latitude_list[i]), float(longitude_list[i])))

    date_occurence_minutes_array = time_obtain_list(date_occurence_array, True)
    date_reported_minutes_array = time_obtain_list(date_reported_array, False)

    # SORTING BY TIME
    time_sort_indices = np.argsort(date_occurence_minutes_array)
    date_occurence_minutes_array = np.array(date_occurence_minutes_array)[time_sort_indices]
    date_reported_minutes_array = np.array(date_reported_minutes_array)[time_sort_indices]
    latitude_list = latitude_list[time_sort_indices]
    longitude_list = longitude_list[time_sort_indices]
    country_array = country_array[time_sort_indices]
    city_array = city_array[time_sort_indices]
    state_array = state_array[time_sort_indices]
    ufo_shape_array = ufo_shape_array[time_sort_indices]
    date_occurence_array = date_occurence_array[time_sort_indices]
    date_reported_array = date_reported_array[time_sort_indices]
    ufo_description_array = ufo_description_array[time_sort_indices]

    # Recreate the derived lists with sorted data
    lat_long_tuple_list = [(float(latitude_list[i]), float(longitude_list[i])) 
                        for i in range(len(latitude_list))]
    csc_list = [(city_array[i], state_array[i], country_array[i]) 
                for i in range(len(city_array))]

    ufo_list = []
    ufo_list.append(csc_list)
    ufo_list.append(date_occurence_minutes_array)
    ufo_list.append(date_reported_minutes_array)
    ufo_list.append(lat_long_tuple_list)
    ufo_list.append(ufo_shape_array)
    ufo_list.append(ufo_description_array)
    ufo_list.append(date_occurence_array)
    ufo_list.append(date_reported_array)

    return ufo_list

def network_creation_function(file_name, distance_threshold, network_percentage, time_step):

    # network_percentage is a float between (0, 1) that designates the fraction of the network I want to be generated, 0.5 means 50 percent 
    csc_list, date_occurence_minutes_array, date_reported_minutes_array, lat_long_tuple_list, ufo_shape_array, ufo_description_array, date_occurence_array, date_reported_array = data_acquisition(file_name)

    ufo_network = nx.Graph()

    k_avg_list = []
    average_clustering_coefficient_list = []
    time_snapshots = []

    start_time = date_occurence_minutes_array[0]
    end_time = date_occurence_minutes_array[-1]
    
    fraction_end_time = start_time + int(network_percentage*(end_time - start_time))
    time_array = np.arange(start_time, fraction_end_time + time_step, time_step)

    indices_in_order = []
    for t in range(len(time_array) - 1):
        t_i = time_array[t]; t_f = time_array[t+1]
        indices = [k for k in range(len(date_occurence_minutes_array)) if t_i < date_occurence_minutes_array[k] <= t_f ]

        indices_in_order.append(indices)

    added_nodes = []  
    for k, indices in enumerate(indices_in_order):

        time_snapshot = time_array[k + 1]

        for i in indices:

            node = int(i)

            ufo_network.add_node(node,
                                 lat_long_tuple=lat_long_tuple_list[i],
                                 city_state_country_tuple=csc_list[i],
                                 shape=ufo_shape_array[i],
                                 date_occured_minutes=date_occurence_minutes_array[i],
                                 date_reported_minutes=date_reported_minutes_array[i],
                                 date_occurred=date_occurence_array[i],
                                 date_reported=date_reported_array[i],
                                 time_minutes=date_occurence_minutes_array[i],
                                 description=ufo_description_array[i])

            for prev_node in added_nodes:
                coord_i = lat_long_tuple_list[i]
                coord_j = lat_long_tuple_list[prev_node]

                distance = compute_haversine(coord_i, coord_j)

                if distance <= distance_threshold:
                    normalized_weight = distance / distance_threshold
                    ufo_network.add_edge(node, prev_node, weight=normalized_weight, distance=distance)

            added_nodes.append(node)

        if ufo_network.number_of_nodes() > 0:
            k_avg_list.append(np.mean([d for _, d in ufo_network.degree()]))

            if ufo_network.number_of_nodes() > 1:
                average_clustering_coefficient_list.append(np.mean(list(nx.clustering(ufo_network).values())))
            else:
                average_clustering_coefficient_list.append(0.0)
            time_snapshots.append(time_snapshot)

    print(f"\nFinal total size of network: {ufo_network.number_of_nodes()} nodes, {ufo_network.number_of_edges()} edges")

    return ufo_network, k_avg_list, average_clustering_coefficient_list, time_snapshots

def network_plotter(ufo_network):
    fig = plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-whitegrid')

    edges = ufo_network.edges()
    weights = [ufo_network[u][v]['weight'] for u, v in edges]
    pos = nx.spring_layout(ufo_network)
    nx.draw_networkx_nodes(ufo_network, pos, node_size = 3, node_color='black')
    for (i, j), alpha in zip(edges, weights):
        nx.draw_networkx_edges(ufo_network, pos, [(i, j)], alpha=alpha, width=2)
    plt.title('fig 1. spring diagram of the temporal UFO network')
    plt.tight_layout()
    plt.savefig('spring_diagram.png', dpi = 400)
    plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.style.use('seaborn-v0_8-whitegrid')
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180)
    m.fillcontinents(color='limegreen',lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')

    for index in list(ufo_network.nodes()):
        xpt,ypt = m(ufo_network.nodes[index]['lat_long_tuple'][1], ufo_network.nodes[index]['lat_long_tuple'][0])  
        m.plot(xpt,ypt,'ko', markersize = 2)

    for init_node, final_node in ufo_network.edges():
        xpt_i, ypt_i = m(ufo_network.nodes[init_node]['lat_long_tuple'][1], 
                        ufo_network.nodes[init_node]['lat_long_tuple'][0])
        xpt_f, ypt_f = m(ufo_network.nodes[final_node]['lat_long_tuple'][1], 
                        ufo_network.nodes[final_node]['lat_long_tuple'][0])
        plt.plot([xpt_i, xpt_f], [ypt_i, ypt_f], 'k-', alpha = ufo_network[init_node][final_node]["weight"])


    plt.title("fig 2. geographical map of the temporal UFO network")
    plt.ylim(23, 52)
    plt.xlim(-127, -67)
    plt.xticks(np.arange(-125, -65, 10), labels=[f'{abs(x)}°W' for x in np.arange(-125, -65, 10)])
    plt.yticks(np.arange(25, 55, 5), labels=[f'{y}°N' for y in np.arange(25, 55, 5)])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()

def network_characteristics(ufo_network):
    C = components(ufo_network)
    print("The number of connected components is:", len(C))

    C = sorted(C, key=lambda c: len(c), reverse=True)
    component_sizes = [len(c) for c in C]

    W = nx.adjacency_matrix(ufo_network).toarray()
    A = (W != 0).astype(int)

    s_array = np.sum(A * W, axis=1)
    clustering_coefficient_list = [i for i in nx.clustering(ufo_network).values()]
    mean_clustering_coefficient = np.mean(clustering_coefficient_list)
    print(f"The node with the highest clustering coefficient is node {clustering_coefficient_list.index(max(clustering_coefficient_list))}, with a clustering coefficient of {max(clustering_coefficient_list)}")
    print(f"The average clustering coefficient is: {np.mean(clustering_coefficient_list)}")

def weight_distribution(ufo_network):
# Get 20 logarithmically spaced bins between kmin and kmax
    weight_list = []
    for init_node_num in ufo_network:
        for final_node_num in ufo_network:
            if init_node_num != final_node_num and ufo_network.has_edge(init_node_num, final_node_num) == True:
                if ufo_network.edges[init_node_num, final_node_num]['weight'] != np.nan:
                # if ufo_network.has_edge(init_node_num, final_node_num) == True:
                    weight_list.append(ufo_network.edges[init_node_num, final_node_num]['weight'])

    num_bins = 20
    weight_bin_edges = np.linspace(min(weight_list), max(weight_list), num=num_bins)

    # histogram the data into these bins
    weight_density, _ = np.histogram(weight_list, bins=weight_bin_edges, density=True)
    plt.figure(figsize = (8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # "x" should be midpoint (IN LOG SPACE) of each bin
    log_be = np.log10(weight_bin_edges)
    x = 10**((log_be[1:] + log_be[:-1])/2)

    plt.plot(x, weight_density, 'ko')
    plt.xlabel(f"$w$")
    plt.ylabel(f"$p(w)$")
    plt.title("fig 3. weight distribution of temporal UFO network")
    plt.tight_layout()
    plt.show()

def main():
    distance_threshold = 100 # km
    network_end = 1
    time_step = 10000
    ufo_network, k_avg_list, avg_clustering_list, times_list = network_creation_function("ufo_sightings_cleaned.csv", distance_threshold, network_end, time_step)
    # plt.plot(times_list, k_avg_list)
    # plt.plot(times_list, avg_clustering_list)
    # plt.show()
    network_plotter(ufo_network)
    # weight_distribution(ufo_network)
if __name__ == "__main__":
    main()


