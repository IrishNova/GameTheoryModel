#!/usr/bin/env python3
"""
IBKR Data Collector for Trade Simulation Model
Run this script separately from command line, not in Jupyter
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import time
import os


class FXDataCollector:
    def __init__(self, port=4001, client_id=1):
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    def connect(self):
        """Connect to IB Gateway"""
        try:
            self.ib.connect('127.0.0.1', self.port, clientId=self.client_id)
            print(f"✓ Connected to IB Gateway on port {self.port}")
            # Handle version info more carefully
            try:
                if hasattr(self.ib, 'serverVersion'):
                    print(f"  Server version: {self.ib.serverVersion()}")
                elif hasattr(self.ib.client, 'serverVersion'):
                    print(f"  Server version: {self.ib.client.serverVersion()}")
            except:
                print("  Connected (version info not available)")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def get_fx_pairs(self):
        """Define currency pairs for the model - using IBKR conventions"""
        # IBKR convention: USD is usually the base currency except for EUR, GBP, AUD, NZD
        pairs = [
            'EURUSD',  # EUR as base (IBKR convention)
            'USDCNH',  # USD as base - using CNH (offshore yuan) instead of CNY
            'USDJPY',  # USD as base
            'USDCAD',  # USD as base
            'USDMXN',  # USD as base
            'GBPUSD',  # GBP as base (IBKR convention)
            'USDSGD'  # USD as base
        ]
        return pairs

    def collect_historical_data(self, pair, duration='2 Y', bar_size='1 day'):
        """Collect historical data for a currency pair"""
        try:
            contract = Forex(pair)
            self.ib.qualifyContracts(contract)

            # Check if contract was qualified
            if not contract.conId:
                print(f"  ✗ {pair}: Contract not available on IBKR")
                return pd.DataFrame()

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='MIDPOINT',
                useRTH=True,
                formatDate=1
            )

            df = util.df(bars)
            if not df.empty:
                df.set_index('date', inplace=True)
                print(f"  ✓ {pair}: Retrieved {len(df)} bars")
                return df
            else:
                print(f"  ✗ {pair}: No data retrieved")
                return pd.DataFrame()

        except Exception as e:
            print(f"  ✗ {pair}: Error - {str(e)[:100]}")
            return pd.DataFrame()
        finally:
            # Rate limiting
            time.sleep(2)

    def calculate_volatility(self, df, window=30):
        """Calculate annualized volatility"""
        if df.empty:
            return None

        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)

        return {
            'average': float(df['volatility'].mean()),
            'current': float(df['volatility'].iloc[-1]) if not pd.isna(df['volatility'].iloc[-1]) else None,
            'min': float(df['volatility'].min()),
            'max': float(df['volatility'].max())
        }

    def run_collection(self):
        """Main collection process"""
        # Create data directory if it doesn't exist
        data_dir = "/Users/ryanmoloney/Desktop/DePaul 24/GameTheoryModel/data/historical"
        os.makedirs(data_dir, exist_ok=True)

        print("\n=== Starting FX Data Collection ===")
        print(f"Data will be saved to: {data_dir}")

        if not self.connect():
            print("Failed to connect. Exiting.")
            return False

        pairs = self.get_fx_pairs()
        print(f"\nCollecting data for {len(pairs)} currency pairs...")

        # Collect historical data
        historical_data = {}
        standardized_data = {}  # Store as XXX/USD format

        for pair in pairs:
            df = self.collect_historical_data(pair)
            if not df.empty:
                historical_data[pair] = df

                # Standardize to XXX/USD format
                if pair.startswith('USD'):
                    # Invert the rate (e.g., USDJPY -> JPYUSD)
                    new_pair = pair[3:] + 'USD'
                    standardized_data[new_pair] = df.copy()
                    standardized_data[new_pair]['close'] = 1 / df['close']
                    standardized_data[new_pair]['open'] = 1 / df['open']
                    standardized_data[new_pair]['high'] = 1 / df['low']  # Note: inverted
                    standardized_data[new_pair]['low'] = 1 / df['high']  # Note: inverted
                else:
                    # Already in correct format (EURUSD, GBPUSD)
                    standardized_data[pair] = df

        print(f"\nSuccessfully collected data for {len(historical_data)} pairs")

        # Calculate volatilities using standardized data
        volatilities = {}
        for pair, df in standardized_data.items():
            vol = self.calculate_volatility(df)
            if vol:
                volatilities[pair] = vol

        # Calculate correlation matrix using standardized data
        print("\nCalculating correlation matrix...")
        close_prices = pd.DataFrame()
        for pair, df in standardized_data.items():
            if not df.empty:
                close_prices[pair] = df['close']

        returns = close_prices.pct_change().dropna()
        correlation_matrix = returns.corr()

        # Prepare export data
        export_data = {
            'metadata': {
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': 'IBKR',
                'port': self.port,
                'pairs_collected': len(standardized_data),
                'original_pairs': list(historical_data.keys()),
                'standardized_pairs': list(standardized_data.keys())
            },
            'volatilities': volatilities,
            'correlations': correlation_matrix.to_dict(),
            'current_rates': {}
        }

        # Add current rates (standardized)
        for pair, df in standardized_data.items():
            if not df.empty:
                export_data['current_rates'][pair] = float(df['close'].iloc[-1])

        # Save to JSON in data directory
        output_file = os.path.join(data_dir, 'fx_model_parameters.json')
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Data exported to {output_file}")

        # Save both raw and standardized data to CSV in data directory
        for pair, df in historical_data.items():
            csv_path = os.path.join(data_dir, f'fx_data_raw_{pair}.csv')
            df.to_csv(csv_path)

        for pair, df in standardized_data.items():
            csv_path = os.path.join(data_dir, f'fx_data_{pair}.csv')
            df.to_csv(csv_path)

        print(f"✓ Raw and standardized data saved to CSV files in {data_dir}")

        # Also save a summary CSV with all current rates
        summary_df = pd.DataFrame({
            'pair': list(export_data['current_rates'].keys()),
            'rate': list(export_data['current_rates'].values()),
            'volatility': [volatilities.get(p, {}).get('average', None) for p in export_data['current_rates'].keys()]
        })
        summary_path = os.path.join(data_dir, 'fx_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved to {summary_path}")

        # Disconnect
        self.ib.disconnect()
        print("\n✓ Disconnected from IB Gateway")

        return True


def main():
    """Main entry point"""
    # You can modify these parameters
    PORT = 4001  # Change if your port is different
    CLIENT_ID = 1

    collector = FXDataCollector(port=PORT, client_id=CLIENT_ID)

    try:
        success = collector.run_collection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if collector.ib.isConnected():
            collector.ib.disconnect()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        if collector.ib.isConnected():
            collector.ib.disconnect()
        sys.exit(1)


if __name__ == "__main__":
    main()