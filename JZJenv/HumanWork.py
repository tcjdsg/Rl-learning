class HumanWork:
    def init(self):
        self.walking = False  # 步行状态：True 表示正在行走
        self.from_pos = 0  # 出发的飞机id
        self.to_pos = 0  # 目的飞机id
        self.walk_start_time = 0  # 出发的时间
        self.walk_distance = 0  # 行走距离
        self.walk_speed = 0.3  # 行走速度
        self.walk_time = 0  # 行走时间