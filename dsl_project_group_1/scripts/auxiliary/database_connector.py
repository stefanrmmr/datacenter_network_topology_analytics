import mariadb
import sys
import pandas as pd


def get_cursor_to_cerberus() -> mariadb.connection.cursor:
    """
    retrieves cursor to cerberus database
    See https://mariadb.com/de/resources/blog/how-to-connect-python-programs-to-mariadb/ for more Information

    Returns: mariadb cursor object

    """
    # Connect to MariaDB Platform
    try:
        conn = mariadb.connect(
            user="dslab_user",
            password="dslab@lkn2021",
            host="10.162.148.20",
            port=3311,
            database="cerberus2"

        )
    except mariadb.Error as e:
        print(f"Error connecting to Cerberus: {e}")
        sys.exit(1)

    # Get Cursor
    return conn.cursor()


def get_df_from_cursor(cursor) -> pd.DataFrame:
    """

    Args:
        cursor: mariadb cursor object

    Returns: Dataframe

    """
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [x[0] for x in cursor.description]
    return df


def get_df_from_table(cursor, table_name, column_names='*') -> pd.DataFrame:
    """

    Args:
        cursor: mariadb cursor object
        table_name: name of the table you want to fetch
        column_names: names of columns you need eg. 'column1, column2, column3', default = all columns

    Returns: Dataframe

    """
    condition = ''
    if table_name == 'simulation':
        condition = ' where simulation_id  <= 1293'
    cursor.execute(f"""Select {column_names} from {table_name}{condition}""")
    return get_df_from_cursor(cursor)


def get_df_from_sql_query(cursor, query) -> pd.DataFrame:
    """

    Args:
        cursor: mariadb cursor object
        query: SQL query

    Returns: Dataframe

    """
    cursor.execute(query)
    return get_df_from_cursor(cursor)


def fetch_random_sample(cursor, table_name, column_names='*', n=10000, random_seed=1):
    """
    Caution: quite inefficient due to full table scan

    Args:
        cursor: mariadb cursor object
        table_name: name of the table you want to fetch
        column_names: names of columns you need eg. 'column1, column2, column3', default = all columns
        n: number of rows
        random_seed: using the same seed will produce same results

    Returns: Dataframe

    """
    cursor.execute(f"""SELECT {column_names} from {table_name} ORDER BY RAND({random_seed}) LIMIT {n}; """)
    return get_df_from_cursor(cursor)


def process_list_for_sql(x) -> str:
    """

    Args:
        x: sequence or array type object

    Returns: string with sql compatible list

    """
    return '(' + str(list(x))[1:-1] + ')'


def get_sample_by_algorithm_and_topology_and_flowgen_ids(cur, algorithm_id: int, topology_id: int, flowgen_id: int,
                                                         table_name: str, column_names='*') -> pd.DataFrame:
    """

    Args:
        cur: mariadb cursor object
        algorithm_id: unique algorithm identifier -> algorithm_id / fk_algorithm_id
        topology_id: unique topology identifier -> topology_id / fk_topology_id
        flowgen_id: unique flow generator identifier -> flowgen_id / fk_flow_generator_id
        table_name: name of the table you want to fetch -> one of [flow, flowcompletiontime, totalflowrate, numberflows]
        column_names: names of columns you need eg. 'column1, column2, column3', default = all columns

    Returns: Dataframe

    """
    simulation_df = get_df_from_table(cur, 'simulation')
    sample = simulation_df[
        (simulation_df['fk_algorithm_id'] == algorithm_id) & (simulation_df['fk_topology_id'] == topology_id) & (
                    simulation_df['fk_flow_generator_id'] == flowgen_id)]
    sample_simulation_ids = process_list_for_sql(sample['simulation_id'])
    if table_name == 'flow':
        query = f"""select fk_flow_id from flowcompletiontime where fk_simulation_id IN {sample_simulation_ids}"""
        flow_ids = process_list_for_sql(get_df_from_sql_query(cur, query)['fk_flow_id'])
        query = f"""select {column_names} from flow where flow_id IN {flow_ids}"""
    else:
        query = f"""select {column_names} from {table_name} where fk_simulation_id IN {sample_simulation_ids}"""
    return get_df_from_sql_query(cur, query)


def get_sample_by_simulation_id(cur, simulation_id: int, table_name: str, column_names='*') -> pd.DataFrame:
    """

    Args:
        cur: mariadb cursor object
        simulation_id: unique simulation identifier -> simulation_id / fk_simulation_id
        table_name: name of the table you want to fetch -> one of [flow, flowcompletiontime, totalflowrate, numberflows]
        column_names: names of columns you need eg. 'column1, column2, column3', default = all columns

    Returns: Dataframe

    """
    if table_name == 'flow':
        query = f"""select fk_flow_id from flowcompletiontime where fk_simulation_id = {simulation_id}"""
        flow_ids = process_list_for_sql(get_df_from_sql_query(cur, query)['fk_flow_id'])
        query = f"""select {column_names} from flow where flow_id IN {flow_ids}"""
    else:
        query = f"""select {column_names} from {table_name} where fk_simulation_id = {simulation_id}"""
    return get_df_from_sql_query(cur, query)
