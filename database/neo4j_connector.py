from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        if self.driver:
            self.driver.close()
        
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
            
    def create_plant(self, name, scientific_name, description):
        query = """
        CREATE (p:Plant {
            name: $name,
            scientific_name: $scientific_name,
            description: $description
        })
        RETURN p
        """
        return self.query(query, 
                         {"name": name, 
                          "scientific_name": scientific_name, 
                          "description": description})
