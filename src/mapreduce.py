from mrjob.job import MRJob
import datetime
import json
from mr3px.csvprotocol import CsvProtocol

class Count(MRJob):

    OUTPUT_PROTOCOL = CsvProtocol

    def mapper(self, _, line):
        data = line.replace(', ', '-').split(",")
        if data[0] != "Mã học viên":
            data[1] = int(
                datetime.datetime.strptime(data[1], "%m/%d/%Y %H:%M").timestamp()
            )
            data[2] = int(
                datetime.datetime.strptime(data[2], "%m/%d/%Y %H:%M").timestamp()
            )
            data[5] = int(data[5]) / 1000000
            data[6] = data[6].replace(" buổi", "")
            data[6] = data[6].replace("tháng", "30")
            data[7] = data[7].replace("Không đạt", "0")
            data[7] = data[7].replace("Đạt", "1")
            if data[7] == "":
                data[7] = 0
            data[8] = data[8].replace("%", "")
            if data[8] == "":
                data[8] = 0
            data[6] = int(data[6])
            data[7] = int(data[7])
            data[8] = int(data[8])
            yield (data[0], ",".join(str(x) for x in data))
        else:
            yield (["HEADER"], ",".join(str(x) for x in data))

    def combiner(self, key, lines):
        datum = [line.split(",") for line in lines]
        if key[0] != "HEADER":
            for i in range(len(datum) - 1):
                for j in range(i + 1, len(datum)):
                    if int(datum[j][2]) < int(datum[i][2]):
                        datum[i], datum[j] = datum[j], datum[i]
            for i in range(len(datum)):
                datum[i].append(i+1)
                if i == len(datum) - 1:
                    datum[i].append(0)
                else:
                    datum[i].append(1)  
                k = int(datum[i][2])
                datum[i][1] = datetime.datetime.fromtimestamp(int(datum[i][1])).strftime('%m/%d/%Y %H:%M')
                datum[i][2] = datetime.datetime.fromtimestamp(int(datum[i][2])).strftime('%m/%d/%Y %H:%M')
                datum[i][5] = float(datum[i][5])
                datum[i][6] = int(datum[i][6])
                datum[i][7] = int(datum[i][7])
                datum[i][8] = int(datum[i][8])              
                yield ([key, k], datum[i])
        else:
            datum[0].append("Lần đăng ký")
            datum[0].append("Có tái tục không")
            yield (["HEADER"], datum[0])

    def reducer(self, key, line):
        for row in line:
            yield (None, row)

if __name__ == "__main__":
    Count.run()
