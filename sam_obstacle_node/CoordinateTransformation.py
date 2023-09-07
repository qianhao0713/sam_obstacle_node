from math import *

class CoordinateTransformation:
    def __init__(self):
        self.a = 6378137.0  # 为椭球长半轴
        self.b = 6356752.31414  # 为椭球短半轴
        self.f = 1 / 298.257223563
        self.e1 = sqrt(self.a * self.a - self.b * self.b) / self.a # 第一偏心率
        self.e2 = sqrt(self.a * self.a - self.b * self.b) / self.b

        self.ori_BLH_point = [38.95390218, 105.68668800, 1446.799]
        self.ori_BLH2XYZ_point = self.BLH2XYZ(self.ori_BLH_point)

    def rad2angle(self, r):
        """
        该函数可以实现弧度到角度的转换.
        :param r:  弧度
        :return:  a, 对应的角度
        """
        a = r * 180.0 / pi
        return a

    def angle2rad(self, a):
        """
        该函数可以实现角度到弧度的转换.
        :param a:  角度
        :return:  r, 对应的弧度
        """
        r = a * pi / 180.0
        return r

    def XYZ2BLH(self, XYZPoint):
        X, Y, Z = XYZPoint  # 空间直角坐标系

        R = sqrt(X * X + Y * Y)
        B0 = atan2(Z, R)

        while 1:
            N = self.a / sqrt(1.0 - self.f * (2 - self.f) * sin(B0) * sin(B0))
            B_ = atan2(Z + N * self.f * (2 - self.f) * sin(B0), R)
            if (fabs(B_ - B0) < 1.0e-7):
                break
            B0 = B_
        L_ = atan2(Y, X)
        H = R / cos(B_) - N
        B = self.rad2angle(B_)
        L = self.rad2angle(L_)
        return B, L, H

    def BLH2XYZ(self, BLHPoint):
        B, L, H = BLHPoint

        B = self.angle2rad(B)  # 角度转为弧度
        L = self.angle2rad(L)  # 角度转为弧度

        N = self.a / sqrt(1 - self.e1 * self.e1 * sin(B) * sin(B))  # 卯酉圈半径, 单位 m

        X = (N + H) * cos(B) * cos(L)
        Y = (N + H) * cos(B) * sin(L)
        Z = (N * (1 - self.e1 * self.e1) + H) * sin(B)
        return X, Y, Z

    def NEU2BLH(self, ENUPoints):
        E, N, U = ENUPoints

        B = self.angle2rad(self.ori_BLH_point[0])
        L = self.angle2rad(self.ori_BLH_point[1])

        dx = -sin(B) * cos(L) * N - sin(L) * E + cos(B) * cos(L) * U
        dy = -sin(B) * sin(L) * N + cos(L) * E + cos(B) * sin(L) * U
        dz = cos(B) * N + 0 + sin(B) * U

        outputPoint_XYZ_x = self.ori_BLH2XYZ_point[0] + dx
        outputPoint_XYZ_y = self.ori_BLH2XYZ_point[1] + dy
        outputPoint_XYZ_z = self.ori_BLH2XYZ_point[2] + dz

        outputPoint_XYZ = [outputPoint_XYZ_x, outputPoint_XYZ_y, outputPoint_XYZ_z]

        B, L, H = self.XYZ2BLH(outputPoint_XYZ)
        outputPoint = [B, L, H]

        return outputPoint


if __name__ == '__main__':
    neu_point = [-1896.31776, 1612.173562, -31.929]
    co_trans = CoordinateTransformation()
    res = co_trans.NEU2BLH(neu_point)
    print(res)











