class PhysicsLoss:
    def __init__(self, w_fid=1.0, w_bw=0.05, w_sm=0.05, w_time=0.0):
        self.w_fid = w_fid
        self.w_bw = w_bw
        self.w_sm = w_sm
        self.w_time = w_time

    def compute(self, m: dict) -> float:
        return (
            - self.w_fid * m["fidelity"]
            + self.w_bw * m["bandwidth"]
            + self.w_sm * m["smoothness"]
            + self.w_time * m["t_reset"]
        )
