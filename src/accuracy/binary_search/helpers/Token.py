class Token:
    def __init__(self, id: int, string: str):
        self.id = id 
        self.string = ''.join(string)
    
    def __str__(self):
        return self.string