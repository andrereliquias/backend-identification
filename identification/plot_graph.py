import io
import base64
import matplotlib.pyplot as plt
import pandas as pd


class PlotGraph:
    def run(self, data):
        # Converte os dados para DataFrame
        data_frame = pd.DataFrame(data)

        # Converte as datas de string para objeto
        data_frame['dateTime'] = pd.to_datetime(data_frame['dateTime'])

        # Define a quantidade de registros por data/hora
        data_frame['count'] = data_frame.groupby('dateTime')['dateTime'].transform('count')

        # Reamostra os dados por hora e preenche as horas que não têm dados com zero
        data_frame.set_index('dateTime', inplace=True)
        data_frame = data_frame.resample('S').sum().fillna(0)

        # Grafico
        fig, ax = plt.subplots()
        data_frame.plot(y='count', ax=ax)

        ax.set_xlabel('Data/Hora')
        ax.set_ylabel('Quantidade')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return {'plot_url': plot_url}

    def empty_graph(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('Data/Hora')
        ax.set_ylabel('Quantidade')
        output = io.BytesIO()
        plt.savefig(output, format='png')
        plt.close(fig)
        output.seek(0)

        plot_url = base64.b64encode(output.getvalue()).decode()

        return {'plot_url': plot_url}
