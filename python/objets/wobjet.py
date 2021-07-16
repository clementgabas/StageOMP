class WDegCarre():
    
    def __init__(self, ra_deg: float, dec_deg: float, cadran: str, exposition: float, seeing: float):
        self.ra = ra_deg
        self.dec = dec_deg
        self.w = cadran
        self.exposition = exposition
        self.seeing = seeing
        
    def __str__(self):
        os = str()
        os += f"RA: {self.ra} (deg)" + "\n"
        os += f"DEC: {self.dec} (deg)" + "\n"
        os += f"Exposition: {self.exposition} (min)" + "\n"
        os += f"Seeing: {self.seeing} (arc sec)" + "\n"
        return os
    
