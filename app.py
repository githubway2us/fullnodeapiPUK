import hashlib
import time
import json
import secrets
import os
import base58
import plyvel
import pickle
from ecdsa import SigningKey, SECP256k1, VerifyingKey
from binascii import hexlify, unhexlify
import socket
import threading
import queue
import sys
import select
import requests

# -------------------------------
# Exchange Request
# -------------------------------
class ExchangeRequest:
    def __init__(self, source_amount, source_chain_id, source_txid, receiver):
        self.source_amount = source_amount
        self.source_chain_id = source_chain_id
        self.source_txid = source_txid
        self.receiver = receiver
        self.amount = self.calculate_puk_amount()
        self.signature = None
        self.pubkey = None
        self.exchange_id = self.compute_exchange_id()

    def calculate_puk_amount(self):
        try:
            oracle = Oracle()
            source_rate = oracle.get_exchange_rate(self.source_chain_id)
            puk_rate = oracle.get_puk_rate()
            puk_amount = (self.source_amount * source_rate) / puk_rate
            return round(puk_amount, 8)
        except Exception as e:
            raise Exception(f"Failed to calculate PUK amount: {e}")

    def compute_exchange_id(self):
        data = json.dumps({
            "source_amount": self.source_amount,
            "source_chain_id": self.source_chain_id,
            "source_txid": self.source_txid,
            "receiver": self.receiver
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def sign(self, private_key):
        sk = SigningKey.from_string(unhexlify(private_key), curve=SECP256k1)
        vk = sk.get_verifying_key()
        self.pubkey = hexlify(vk.to_string()).decode()
        self.signature = hexlify(sk.sign(self.exchange_id.encode())).decode()

    def verify(self):
        if not self.signature or not self.pubkey:
            return False
        vk = VerifyingKey.from_string(unhexlify(self.pubkey), curve=SECP256k1)
        try:
            return vk.verify(unhexlify(self.signature), self.exchange_id.encode())
        except:
            return False

    def to_dict(self):
        return {
            "exchange_id": self.exchange_id,
            "source_amount": self.source_amount,
            "source_chain_id": self.source_chain_id,
            "source_txid": self.source_txid,
            "receiver": self.receiver,
            "amount": self.amount,
            "signature": self.signature,
            "pubkey": self.pubkey
        }

    @classmethod
    def from_dict(cls, data):
        req = cls(data["source_amount"], data["source_chain_id"], data["source_txid"], data["receiver"])
        req.amount = data["amount"]
        req.signature = data["signature"]
        req.pubkey = data["pubkey"]
        return req

# -------------------------------
# Liquidity Pool
# -------------------------------
class LiquidityPool:
    def __init__(self, initial_puk=1_000_000, initial_usd=1_000_000, total_supply=1_000_000_000):
        self.total_supply = total_supply
        self.puk_balance = initial_puk
        self.usd_balance = initial_usd
        self.k = self.puk_balance * self.usd_balance
        self.burn_rate = 0.0001  # 0.01% เพื่อลดการเผา
        self.mint_to_pool(total_supply * 0.6)  # Mint 60% เข้า pool

    def mint_to_pool(self, amount):
        self.puk_balance += amount
        self.k = self.puk_balance * self.usd_balance
        print(f"Minted {amount} PUK to pool. New puk_balance: {self.puk_balance}")

    def burn_from_pool(self):
        burn_amount = self.puk_balance * self.burn_rate
        self.puk_balance -= burn_amount
        self.k = self.puk_balance * self.usd_balance
        if self.k <= 0:
            raise Exception("Invalid pool state: k <= 0")
        print(f"Burned {burn_amount} PUK from pool. New puk_balance: {self.puk_balance}")
        return burn_amount

    def add_burned_to_mining(self, burn_amount):
        return burn_amount

    def get_puk_price(self):
        return self.usd_balance / self.puk_balance if self.puk_balance > 0 else 0

    def add_liquidity(self, puk_amount=0, usd_amount=0):
        self.puk_balance += puk_amount
        self.usd_balance += usd_amount
        self.k = self.puk_balance * self.usd_balance
        if self.k <= 0:
            raise Exception("Invalid pool state: k <= 0")
        print(f"Added {puk_amount} PUK and {usd_amount} USD to pool. New puk_balance: {self.puk_balance}, usd_balance: {self.usd_balance}")

    def remove_liquidity(self, puk_amount=0, usd_amount=0):
        if puk_amount > self.puk_balance or usd_amount > self.usd_balance:
            raise Exception("Insufficient liquidity")
        self.puk_balance -= puk_amount
        self.usd_balance -= usd_amount
        self.k = self.puk_balance * self.usd_balance
        if self.k <= 0:
            raise Exception("Invalid pool state: k <= 0")
        print(f"Removed {puk_amount} PUK and {usd_amount} USD from pool. New puk_balance: {self.puk_balance}, usd_balance: {self.usd_balance}")

    def validate_pool(self):
        if self.puk_balance <= 0 or self.usd_balance <= 0:
            raise Exception("Invalid pool state: balance <= 0")
        if abs(self.puk_balance * self.usd_balance - self.k) > 1e-8:
            raise Exception("Invalid pool state: k invariant broken")
        print("Liquidity pool validated successfully")

    def swap_to_puk(self, usd_amount):
        new_usd_balance = self.usd_balance + usd_amount
        new_puk_balance = self.k / new_usd_balance
        puk_amount = self.puk_balance - new_puk_balance
        self.puk_balance = new_puk_balance
        self.usd_balance = new_usd_balance
        return puk_amount

    def swap_to_usd(self, puk_amount):
        new_puk_balance = self.puk_balance + puk_amount
        new_usd_balance = self.k / new_puk_balance
        usd_amount = self.usd_balance - new_usd_balance
        self.puk_balance = new_puk_balance
        self.usd_balance = new_usd_balance
        return usd_amount

# -------------------------------
# Oracle
# -------------------------------
class Oracle:
    def __init__(self):
        self.pool = LiquidityPool(total_supply=1_000_000_000)
        self.pool.validate_pool()

    def get_exchange_rate(self, source_chain_id, target_currency="USD"):
        mock_rates = {
            "BTC": 116512.04,  # 1 BTC = 116,512.04 USD (19 สิงหาคม 2025)
            "ETH": 3000,
        }
        if source_chain_id not in mock_rates:
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={source_chain_id.lower()}&vs_currencies={target_currency.lower()}"
                response = requests.get(url)
                return response.json()[source_chain_id.lower()][target_currency.lower()]
            except:
                raise Exception(f"Unsupported source chain: {source_chain_id}")
        return mock_rates[source_chain_id]

    def get_puk_rate(self):
        return self.pool.get_puk_price()

    def get_btc_puk_rate(self):
        btc_usd = self.get_exchange_rate("BTC")
        puk_usd = self.get_puk_rate()
        return btc_usd / puk_usd if puk_usd > 0 else 0

# -------------------------------
# Transaction / UTXO
# -------------------------------
class TXOutput:
    def __init__(self, amount, receiver):
        self.amount = amount
        self.receiver = receiver

    def to_dict(self):
        return {"amount": self.amount, "receiver": self.receiver}

    def serialize(self):
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        try:
            return pickle.loads(data)
        except Exception as e:
            print(f"Error deserializing TXOutput: {e}")
            return None

class TXInput:
    def __init__(self, txid, output_index, signature=None, pubkey=None):
        self.txid = txid
        self.output_index = output_index
        self.signature = signature
        self.pubkey = pubkey

    def to_dict(self):
        return {
            "txid": self.txid,
            "output_index": self.output_index,
            "signature": self.signature,
            "pubkey": self.pubkey
        }

class Transaction:
    def __init__(self, inputs, outputs, locktime=0, coinbase_data=None):
        self.inputs = inputs
        self.outputs = outputs
        self.locktime = locktime
        self.coinbase_data = coinbase_data
        self.txid = self.compute_txid()

    def compute_txid(self):
        data = json.dumps({
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "locktime": self.locktime,
            "coinbase_data": self.coinbase_data
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def sign(self, private_key, utxo_set):
        sk = SigningKey.from_string(unhexlify(private_key), curve=SECP256k1)
        vk = sk.get_verifying_key()
        pubkey = hexlify(vk.to_string()).decode()
        data = self._get_signing_data(utxo_set)
        for txi in self.inputs:
            txi.signature = hexlify(sk.sign(data.encode())).decode()
            txi.pubkey = pubkey

    def _get_signing_data(self, utxo_set):
        temp_inputs = [TXInput(txi.txid, txi.output_index) for txi in self.inputs]
        return json.dumps({
            "inputs": [i.to_dict() for i in temp_inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "locktime": self.locktime,
            "coinbase_data": self.coinbase_data
        }, sort_keys=True)

    def verify(self, utxo_set):
        if not self.inputs and not self.coinbase_data:
            return False
        if not self.outputs:
            return False
        total_in = 0
        for txi in self.inputs:
            key = f"{txi.txid}:{txi.output_index}".encode()
            if key not in utxo_set.db and key not in utxo_set.exchange_db:
                return False
            txo = TXOutput.deserialize(utxo_set.db.get(key) or utxo_set.exchange_db.get(key))
            if txo is None:
                return False
            total_in += txo.amount
            if not self._verify_signature(txi, utxo_set):
                return False
        total_out = sum(txo.amount for txo in self.outputs)
        if self.inputs and total_in < total_out:
            return False
        return True

    def _verify_signature(self, txi, utxo_set):
        if not txi.signature or not txi.pubkey:
            return False
        vk = VerifyingKey.from_string(unhexlify(txi.pubkey), curve=SECP256k1)
        data = self._get_signing_data(utxo_set)
        try:
            return vk.verify(unhexlify(txi.signature), data.encode())
        except:
            return False

    def to_dict(self):
        return {
            "txid": self.txid,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "locktime": self.locktime,
            "coinbase_data": self.coinbase_data
        }

    @classmethod
    def from_dict(cls, data):
        inputs = [TXInput(**i) for i in data["inputs"]]
        outputs = [TXOutput(**o) for o in data["outputs"]]
        return cls(inputs, outputs, data["locktime"], data["coinbase_data"])

# -------------------------------
# UTXO Set
# -------------------------------
class UTXOSet:
    def __init__(self):
        self.db = plyvel.DB('utxos', create_if_missing=True)
        self.exchange_db = plyvel.DB('exchange_utxos', create_if_missing=True)

    def update(self, block, chain):
        with self.db.write_batch() as wb:
            for tx in block.transactions:
                print(f"Processing TX: {tx.txid}, Coinbase: {tx.coinbase_data is not None}")
                for txi in tx.inputs:
                    key = f"{txi.txid}:{txi.output_index}".encode()
                    if key in self.db:
                        print(f"Deleting UTXO: {key.decode()}")
                        wb.delete(key)
                    elif key in self.exchange_db:
                        print(f"Deleting Exchange UTXO: {key.decode()}")
                        self.exchange_db.delete(key)
                for idx, txo in enumerate(tx.outputs):
                    key = f"{tx.txid}:{idx}".encode()
                    if tx.coinbase_data and tx.coinbase_data.startswith("exchange"):
                        print(f"Adding Exchange UTXO: {key.decode()}, Amount: {txo.amount}, Receiver: {txo.receiver}")
                        self.exchange_db.put(key, txo.serialize())
                    else:
                        print(f"Adding UTXO: {key.decode()}, Amount: {txo.amount}, Receiver: {txo.receiver}")
                        wb.put(key, txo.serialize())

    def add_exchange_utxo(self, exchange_request, txid, index):
        key = f"{txid}:{index}".encode()
        txo = TXOutput(exchange_request.amount, exchange_request.receiver)
        self.exchange_db.put(key, txo.serialize())
        print(f"Added Exchange UTXO: {key.decode()}, Amount: {txo.amount}, Receiver: {txo.receiver}")

    def verify_exchange_utxo(self, exchange_request):
        for key, _ in self.exchange_db.iterator():
            txo = TXOutput.deserialize(self.exchange_db.get(key))
            if txo is None:
                continue
            if key.decode().startswith(exchange_request.exchange_id):
                return False
        return True

    def get_balance(self, address, chain):
        balance = 0
        utxos = []
        unconfirmed_utxos = []
        try:
            # อ่าน exchange_db
            for key, value in self.exchange_db.iterator():
                txo = TXOutput.deserialize(value)
                if txo is None:
                    print(f"Skipping invalid exchange UTXO: {key.decode()}")
                    continue
                if txo.receiver == address:
                    balance += txo.amount
                    utxos.append(key.decode())
                    print(f"Found exchange UTXO: {key.decode()}, Amount: {txo.amount}")

            # อ่าน db
            for key, value in self.db.iterator():
                txo = TXOutput.deserialize(value)
                if txo is None:
                    print(f"Skipping invalid UTXO: {key.decode()}")
                    continue
                if txo.receiver == address:
                    txid, idx = key.decode().split(":")
                    confirmed = False
                    for block in chain:
                        if any(tx.txid == txid for tx in block.transactions):
                            balance += txo.amount
                            utxos.append(key.decode())
                            print(f"Found confirmed UTXO: {key.decode()}, Amount: {txo.amount}, Block Index: {block.index}, Chain Length: {len(chain)}")
                            confirmed = True
                            break
                    if not confirmed:
                        unconfirmed_utxos.append((key.decode(), txo.amount))
                        print(f"Found unconfirmed UTXO: {key.decode()}, Amount: {txo.amount}")
        except Exception as e:
            print(f"Error calculating balance for {address}: {e}")
        print(f"Confirmed UTXOs for {address}: {utxos}")
        print(f"Unconfirmed UTXOs for {address}: {unconfirmed_utxos}")
        return balance

    def select_utxos(self, address, amount, chain):
        selected = []
        total = 0
        try:
            for key, value in self.exchange_db.iterator():
                txo = TXOutput.deserialize(value)
                if txo is None:
                    print(f"Skipping invalid exchange UTXO: {key.decode()}")
                    continue
                if txo.receiver == address:
                    txid, idx = key.decode().split(":")
                    selected.append(TXInput(txid, int(idx)))
                    total += txo.amount
                    print(f"Selected exchange UTXO: {key.decode()}, Amount: {txo.amount}")
                    if total >= amount:
                        return total, selected
            for key, value in self.db.iterator():
                txo = TXOutput.deserialize(value)
                if txo is None:
                    print(f"Skipping invalid UTXO: {key.decode()}")
                    continue
                if txo.receiver == address:
                    txid, idx = key.decode().split(":")
                    confirmed = False
                    for block in chain:
                        if any(tx.txid == txid for tx in block.transactions):
                            selected.append(TXInput(txid, int(idx)))
                            total += txo.amount
                            print(f"Selected confirmed UTXO: {key.decode()}, Amount: {txo.amount}")
                            if total >= amount:
                                return total, selected
                            confirmed = True
                            break
                    if not confirmed:
                        print(f"Skipping unconfirmed UTXO: {key.decode()}")
        except Exception as e:
            print(f"Error selecting UTXOs for {address}: {e}")
        return total, selected

    def get_size(self):
        count = sum(1 for _ in self.db.iterator()) + sum(1 for _ in self.exchange_db.iterator())
        return count

# -------------------------------
# Merkle Tree
# -------------------------------
class MerkleTree:
    def __init__(self, transactions):
        self.leaves = [tx.txid for tx in transactions]
        self.root = self._build_tree()

    def _build_tree(self):
        if not self.leaves:
            return ""
        level = self.leaves
        while len(level) > 1:
            temp = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                combined = unhexlify(left) + unhexlify(right)
                temp.append(hashlib.sha256(hashlib.sha256(combined).digest()).hexdigest())
            level = temp
        return level[0]

# -------------------------------
# Block / Blockchain
# -------------------------------
def sha256d(s):
    return hashlib.sha256(hashlib.sha256(s).digest()).hexdigest()

class Block:
    def __init__(self, index, transactions, previous_hash, bits=4):
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = int(time.time())
        self.bits = bits
        self.nonce = 0
        self.merkle_root = MerkleTree(transactions).root
        self.hash = self.compute_hash()

    def compute_hash(self):
        data = f"{self.index}{self.merkle_root}{self.previous_hash}{self.timestamp}{self.bits}{self.nonce}"
        return sha256d(data.encode())

    def mine(self):
        target = 2 ** (256 - self.bits * 4)
        max_nonce = 2 ** 32
        for nonce in range(max_nonce):
            self.nonce = nonce
            self.hash = self.compute_hash()
            if int(self.hash, 16) < target:
                return self.hash
        raise Exception("Failed to mine block")

    def serialize(self):
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)

    def to_dict(self):
        return {
            "index": self.index,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "bits": self.bits,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "hash": self.hash
        }

    @classmethod
    def from_dict(cls, data):
        transactions = [Transaction.from_dict(tx) for tx in data["transactions"]]
        block = cls(data["index"], transactions, data["previous_hash"], data["bits"])
        block.timestamp = data["timestamp"]
        block.nonce = data["nonce"]
        block.merkle_root = data["merkle_root"]
        block.hash = data["hash"]
        return block

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.mining_reward = 50
        self.difficulty_adjustment_interval = 10
        self.target_block_time = 10
        self.max_txs_per_block = 1000
        self.burn_rate = 0.1  # เผาค่าธรรมเนียม 10%
        self.total_supply = 1_000_000_000  # อุปทานทั้งหมด 1 พันล้าน PUK
        self.minted_coins = 0
        self.load()
        if not self.chain:
            self.create_genesis_block()
        self.validate_chain()

    def create_genesis_block(self):
        genesis = Block(0, [], "0" * 64, bits=4)
        genesis.mine()
        self.chain.append(genesis)
        self.minted_coins = 0
        self.save()

    def validate_chain(self):
        txids = set()
        for i, block in enumerate(self.chain):
            if i > 0:
                current = block
                prev = self.chain[i - 1]
                if current.previous_hash != prev.hash:
                    raise Exception(f"Invalid chain: Block {i} has invalid previous_hash")
                if current.hash != current.compute_hash():
                    raise Exception(f"Invalid chain: Block {i} has invalid hash")
                if current.merkle_root != MerkleTree(current.transactions).root:
                    raise Exception(f"Invalid chain: Block {i} has invalid Merkle root")
            for tx in block.transactions:
                if tx.txid in txids:
                    raise Exception(f"Invalid chain: Duplicate TXID {tx.txid} in block {i}")
                txids.add(tx.txid)
        print("Blockchain validated successfully")

    def adjust_difficulty(self):
        if len(self.chain) % self.difficulty_adjustment_interval == 0 and len(self.chain) > 0:
            last_adjustment_block = self.chain[-self.difficulty_adjustment_interval]
            time_taken = self.chain[-1].timestamp - last_adjustment_block.timestamp
            expected_time = self.target_block_time * self.difficulty_adjustment_interval
            ratio = time_taken / expected_time
            tx_count = sum(len(block.transactions) for block in self.chain[-self.difficulty_adjustment_interval:])
            avg_txs_per_block = tx_count / self.difficulty_adjustment_interval
            tx_factor = max(1.0, avg_txs_per_block / 100)
            new_bits = self.chain[-1].bits
            if ratio < 0.75:
                new_bits += 1
            elif ratio > 1.25:
                new_bits = max(1, new_bits - 1)
            if self.minted_coins >= 0.4 * self.total_supply:
                new_bits += 2
                print(f"Increasing difficulty: minted {self.minted_coins} PUK (>= 40% of total supply)")
            return min(max(4, new_bits * tx_factor), 20)
        return self.chain[-1].bits

    def add_transaction(self, tx, utxo_set):
        if not tx.verify(utxo_set):
            raise Exception("Invalid transaction")
        if len(self.pending_transactions) < self.max_txs_per_block:
            self.pending_transactions.append(tx)
        else:
            raise Exception("Pending transactions queue full")

    def add_exchange_transaction(self, exchange_request, utxo_set):
        if not exchange_request.verify():
            raise Exception("Invalid exchange request signature")
        if not utxo_set.verify_exchange_utxo(exchange_request):
            raise Exception("Exchange request already used")
        tx = Transaction([], [TXOutput(exchange_request.amount, exchange_request.receiver)],
                        coinbase_data=f"exchange:{exchange_request.exchange_id}")
        tx.txid = tx.compute_txid()
        utxo_set.add_exchange_utxo(exchange_request, tx.txid, 0)
        self.pending_transactions.append(tx)
        print(f"Added exchange transaction: {tx.txid} for {exchange_request.amount} PUK")
        return tx

    def mine_pending(self, miner_address, utxo_set, oracle):
        fees = 0
        burn_amount = 0
        for tx in self.pending_transactions:
            if tx.inputs:
                total_in = sum(TXOutput.deserialize(utxo_set.db.get(f"{txi.txid}:{txi.output_index}".encode()) or
                                          utxo_set.exchange_db.get(f"{txi.txid}:{txi.output_index}".encode())).amount for txi in tx.inputs)
                total_out = sum(txo.amount for txo in tx.outputs)
                fee = total_in - total_out
                fees += fee
                burn_amount += fee * self.burn_rate
        pool_burn = oracle.pool.burn_from_pool()
        additional_reward = oracle.pool.add_burned_to_mining(pool_burn)
        coinbase = Transaction([], [TXOutput(self.mining_reward + fees - burn_amount + additional_reward, miner_address)],
                              coinbase_data=str(time.time()))
        transactions = [coinbase] + self.pending_transactions[:self.max_txs_per_block]
        if not transactions:
            raise Exception("No valid transactions to mine")
        txids = [tx.txid for tx in transactions]
        if len(txids) != len(set(txids)):
            raise Exception("Duplicate transaction IDs detected in block")
        bits = self.adjust_difficulty()
        block = Block(len(self.chain), transactions, self.chain[-1].hash, bits)
        block.mine()
        if block.previous_hash != self.chain[-1].hash:
            raise Exception("Block does not connect to the previous block")
        self.chain.append(block)
        utxo_set.update(block, self.chain)
        self.minted_coins += self.mining_reward + additional_reward
        self.pending_transactions = self.pending_transactions[len(transactions)-1:]
        self.save()
        if burn_amount > 0:
            print(f"Burned {burn_amount} PUK from transaction fees")
        if pool_burn > 0:
            print(f"Added {additional_reward} PUK from pool burn to mining reward")
        if len(self.chain) % 1000 == 0:
            self.validate_chain()
        else:
            print("Blockchain validated successfully")
        return block

    def save(self):
        db = plyvel.DB('blockchain', create_if_missing=True)
        with db.write_batch() as wb:
            for i, block in enumerate(self.chain):
                wb.put(str(i).encode(), block.serialize())
            wb.put(b"minted_coins", str(self.minted_coins).encode())
        db.close()

    def load(self):
        if os.path.exists("blockchain"):
            db = plyvel.DB('blockchain')
            self.chain = []
            i = 0
            while True:
                block_data = db.get(str(i).encode())
                if block_data is None:
                    break
                try:
                    self.chain.append(Block.deserialize(block_data))
                except Exception as e:
                    print(f"Error deserializing block {i}: {e}")
                    break
                i += 1
            minted_coins_data = db.get(b"minted_coins")
            if minted_coins_data:
                self.minted_coins = float(minted_coins_data.decode())
            db.close()

    def to_dict(self):
        return [{
            "index": b.index,
            "previous_hash": b.previous_hash,
            "timestamp": b.timestamp,
            "nonce": b.nonce,
            "bits": b.bits,
            "merkle_root": b.merkle_root,
            "hash": b.hash,
            "transactions": [tx.to_dict() for tx in b.transactions]
        } for b in self.chain]

    def get_total_minted_coins(self):
        return self.minted_coins

# -------------------------------
# Wallet
# -------------------------------
def create_wallet():
    sk = SigningKey.generate(curve=SECP256k1)
    private_key = sk.to_string()
    private_key_hex = hexlify(private_key).decode()
    vk = sk.get_verifying_key()
    public_key = vk.to_string()
    pubkey_hash = hashlib.new('ripemd160')
    pubkey_hash.update(hashlib.sha256(public_key).digest())
    pubkey_hash_bytes = pubkey_hash.digest()
    versioned_payload = b'\x00' + pubkey_hash_bytes
    checksum = sha256d(versioned_payload)[:8]
    checksum_bytes = unhexlify(checksum)
    address = base58.b58encode(versioned_payload + checksum_bytes).decode()
    versioned_privkey = b'\x80' + private_key + b'\x01'
    privkey_checksum = sha256d(versioned_privkey)[:8]
    privkey_checksum_bytes = unhexlify(privkey_checksum)
    wif = base58.b58encode(versioned_privkey + privkey_checksum_bytes).decode()
    return address, private_key_hex, wif

def recover_wallet(private_key_input):
    try:
        if private_key_input.startswith('5') or private_key_input.startswith('K') or private_key_input.startswith('L'):
            decoded = base58.b58decode(private_key_input)
            if len(decoded) != 38:
                raise Exception("Invalid WIF length")
            version, private_key, flag, checksum = decoded[0:1], decoded[1:33], decoded[33:34], decoded[34:]
            computed_checksum = sha256d(version + private_key + flag)[:8]
            if computed_checksum != checksum.hex():
                raise Exception("Invalid WIF checksum")
            if version != b'\x80' or flag != b'\x01':
                raise Exception("Invalid WIF format")
            private_key_hex = private_key.hex()
        else:
            private_key_hex = private_key_input
            private_key = unhexlify(private_key_hex)
            if len(private_key) != 32:
                raise Exception("Invalid private key length")

        sk = SigningKey.from_string(unhexlify(private_key_hex), curve=SECP256k1)
        vk = sk.get_verifying_key()
        public_key = vk.to_string()
        pubkey_hash = hashlib.new('ripemd160')
        pubkey_hash.update(hashlib.sha256(public_key).digest())
        pubkey_hash_bytes = pubkey_hash.digest()
        versioned_payload = b'\x00' + pubkey_hash_bytes
        checksum = sha256d(versioned_payload)[:8]
        checksum_bytes = unhexlify(checksum)
        address = base58.b58encode(versioned_payload + checksum_bytes).decode()
        versioned_privkey = b'\x80' + private_key + b'\x01'
        privkey_checksum = sha256d(versioned_privkey)[:8]
        privkey_checksum_bytes = unhexlify(privkey_checksum)
        wif = base58.b58encode(versioned_privkey + privkey_checksum_bytes).decode()
        return address, private_key_hex, wif
    except Exception as e:
        raise Exception(f"Failed to recover wallet: {e}")

# -------------------------------
# Wallet Storage
# -------------------------------
def save_wallets(wallets):
    db = plyvel.DB('wallets', create_if_missing=True)
    with db.write_batch() as wb:
        for addr, pk_hex in wallets.items():
            wb.put(addr.encode(), pk_hex.encode())
    db.close()

def load_wallets():
    wallets = {}
    if os.path.exists("wallets"):
        db = plyvel.DB('wallets')
        for addr, pk_hex in db:
            wallets[addr.decode()] = pk_hex.decode()
        db.close()
    return wallets

# -------------------------------
# P2P Network
# -------------------------------
class P2PNode:
    def __init__(self, host, port, blockchain, utxo_set):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.utxo_set = utxo_set
        self.peers = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.message_queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()
        self.message_count = 0
        self.last_reset_time = time.time()

    def start(self):
        self.running = True
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"P2P Node started on {self.host}:{self.port}")
        threading.Thread(target=self.accept_connections, daemon=True).start()
        threading.Thread(target=self.process_messages, daemon=True).start()

    def stop(self):
        self.running = False
        self.server_socket.close()
        for peer in self.peers[:]:
            try:
                peer.close()
            except:
                pass
        self.peers = []

    def connect_to_peer(self, peer_host, peer_port):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((peer_host, peer_port))
            self.peers.append(client_socket)
            print(f"Connected to peer {peer_host}:{peer_port}")
            threading.Thread(target=self.handle_peer, args=(client_socket,), daemon=True).start()
            self.request_blockchain(client_socket)
        except Exception as e:
            print(f"Failed to connect to peer {peer_host}:{peer_port}: {e}")

    def accept_connections(self):
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, addr = self.server_socket.accept()
                self.peers.append(client_socket)
                print(f"Accepted connection from {addr}")
                threading.Thread(target=self.handle_peer, args=(client_socket,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def handle_peer(self, client_socket):
        while self.running:
            try:
                client_socket.settimeout(1.0)
                data = client_socket.recv(4096).decode()
                if not data:
                    self.peers.remove(client_socket)
                    client_socket.close()
                    break
                messages = data.strip().split("\n")
                with self.lock:
                    self.message_count += len(messages)
                for message in messages:
                    if message:
                        self.message_queue.put((client_socket, json.loads(message)))
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error handling peer: {e}")
                self.peers.remove(client_socket)
                client_socket.close()
                break

    def process_messages(self):
        while self.running:
            try:
                client_socket, message = self.message_queue.get(timeout=1)
                msg_type = message.get("type")
                if msg_type == "transaction":
                    self.handle_transaction(message["data"])
                elif msg_type == "block":
                    self.handle_block(message["data"])
                elif msg_type == "blockchain_request":
                    self.send_blockchain(client_socket)
                elif msg_type == "blockchain_response":
                    self.handle_blockchain(message["data"])
                elif msg_type == "exchange":
                    self.handle_exchange(message["data"])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")

    def broadcast_transaction(self, tx):
        message = json.dumps({"type": "transaction", "data": tx.to_dict()}) + "\n"
        with self.lock:
            self.message_count += len(self.peers)
        for peer in self.peers[:]:
            try:
                peer.send(message.encode())
            except Exception as e:
                print(f"Error broadcasting transaction to peer: {e}")
                self.peers.remove(peer)
                peer.close()

    def broadcast_block(self, block):
        message = json.dumps({"type": "block", "data": block.to_dict()}) + "\n"
        with self.lock:
            self.message_count += len(self.peers)
        for peer in self.peers[:]:
            try:
                peer.send(message.encode())
            except Exception as e:
                print(f"Error broadcasting block to peer: {e}")
                self.peers.remove(peer)
                peer.close()

    def broadcast_exchange(self, exchange_request):
        message = json.dumps({"type": "exchange", "data": exchange_request.to_dict()}) + "\n"
        with self.lock:
            self.message_count += len(self.peers)
        for peer in self.peers[:]:
            try:
                peer.send(message.encode())
            except Exception as e:
                print(f"Error broadcasting exchange request to peer: {e}")
                self.peers.remove(peer)
                peer.close()

    def request_blockchain(self, client_socket):
        message = json.dumps({"type": "blockchain_request"}) + "\n"
        try:
            client_socket.send(message.encode())
            with self.lock:
                self.message_count += 1
        except Exception as e:
            print(f"Error requesting blockchain: {e}")

    def send_blockchain(self, client_socket):
        message = json.dumps({"type": "blockchain_response", "data": self.blockchain.to_dict()}) + "\n"
        try:
            client_socket.send(message.encode())
            with self.lock:
                self.message_count += 1
        except Exception as e:
            print(f"Error sending blockchain: {e}")

    def handle_transaction(self, tx_data):
        with self.lock:
            tx = Transaction.from_dict(tx_data)
            try:
                self.blockchain.add_transaction(tx, self.utxo_set)
                print(f"Received and added transaction: {tx.txid}")
            except Exception as e:
                print(f"Invalid transaction received: {e}")

    def handle_block(self, block_data):
        with self.lock:
            block = Block.from_dict(block_data)
            if block.previous_hash == self.blockchain.chain[-1].hash:
                if block.hash == block.compute_hash() and block.merkle_root == MerkleTree(block.transactions).root:
                    try:
                        for tx in block.transactions:
                            if not tx.verify(self.utxo_set):
                                print(f"Invalid transaction in block {block.index}")
                                return
                        self.blockchain.chain.append(block)
                        self.utxo_set.update(block, self.blockchain.chain)
                        self.blockchain.pending_transactions = [tx for tx in self.blockchain.pending_transactions if tx.txid not in [t.txid for t in block.transactions]]
                        self.blockchain.save()
                        print(f"Received and added block {block.index}")
                    except Exception as e:
                        print(f"Error adding block {block.index}: {e}")
                else:
                    print(f"Invalid block {block.index}: hash or merkle root mismatch")
            else:
                print(f"Block {block.index} does not connect to current chain")
                self.request_blockchain_from_peers()

    def handle_exchange(self, exchange_data):
        with self.lock:
            exchange_request = ExchangeRequest.from_dict(exchange_data)
            try:
                tx = self.blockchain.add_exchange_transaction(exchange_request, self.utxo_set)
                self.broadcast_transaction(tx)
                print(f"Received and processed exchange request: {exchange_request.exchange_id}")
            except Exception as e:
                print(f"Invalid exchange request received: {e}")

    def request_blockchain_from_peers(self):
        for peer in self.peers[:]:
            self.request_blockchain(peer)

    def get_network_speed(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_reset_time
            if elapsed >= 10:
                speed = self.message_count / max(elapsed, 1)
                self.message_count = 0
                self.last_reset_time = current_time
                return speed
            return self.message_count / max(elapsed, 1)

# -------------------------------
# CLI
# -------------------------------
def clear_input_buffer():
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    elif sys.platform.startswith('win'):
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()

def display_status(p2p_node, blockchain, utxo_set, oracle, wallets):
    puk_price = oracle.get_puk_rate()
    btc_puk_rate = oracle.get_btc_puk_rate()
    print("\n=== PUK Blockchain Status ===")
    print(f"Connected Peers: {len(p2p_node.peers)}")
    print(f"Network Speed: {p2p_node.get_network_speed():.2f} messages/second")
    print(f"Blockchain Height: {len(blockchain.chain)} blocks")
    print(f"Pending Transactions: {len(blockchain.pending_transactions)}")
    print(f"UTXO Set Size: {utxo_set.get_size()} entries")
    print(f"Total Minted Coins: {blockchain.get_total_minted_coins()} PUK ({blockchain.minted_coins / blockchain.total_supply * 100:.2f}% of total supply)")
    print(f"PUK in Liquidity Pool: {oracle.pool.puk_balance} PUK")
    print(f"Current PUK Price: {puk_price:.8f} USD")
    print(f"1 BTC = {btc_puk_rate:.8f} PUK")
    print("\n=== Wallet Balances (Up to 5 Wallets) ===")
    if wallets:
        wallet_list = list(wallets.keys())[:5]
        for addr in wallet_list:
            try:
                balance = utxo_set.get_balance(addr, blockchain.chain)
                print(f"Address: {addr}, Balance: {balance} PUK")
            except Exception as e:
                print(f"Error retrieving balance for {addr}: {e}")
        if len(wallets) > 5:
            print(f"...and {len(wallets) - 5} more wallets. Use option 2 to check specific balances.")
    else:
        print("No wallets found. Creating a new wallet...")
    print("\nPress Enter to access the menu...")

def show_menu():
    print("\n=== PUK Blockchain CLI ===")
    print("1. Create New Wallet")
    print("2. Check Balance")
    print("3. Send Transaction")
    print("4. Mine Block")
    print("5. View Blockchain")
    print("6. Validate Chain")
    print("7. Import New Wallet")
    print("8. Recover Wallet")
    print("9. Connect to Peer")
    print("10. Exit")
    print("11. Check Total Minted Coins")
    print("12. Import Coins from External Chain")
    print("13. Manage Liquidity Pool")

def main():
    blockchain = Blockchain()
    utxo_set = UTXOSet()
    wallets = load_wallets()
    oracle = Oracle()
    p2p_node = P2PNode("localhost", 5000, blockchain, utxo_set)
    p2p_node.start()

    if not wallets:
        print("No wallets found. Creating a new wallet...")
        addr, pk_hex, wif = create_wallet()
        wallets[addr] = pk_hex
        save_wallets(wallets)
        print(f"Created new wallet - Address: {addr}, Private Key (Hex): {pk_hex}, WIF: {wif}")
        print("Please store the Private Key and WIF securely. Do not share them!")

    while True:
        clear_input_buffer()
        try:
            display_status(p2p_node, blockchain, utxo_set, oracle, wallets)
            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
            if rlist:
                input("Press Enter to access the menu...")
                show_menu()
                choice = input("Select option: ").strip().lower()
                if not choice:
                    continue

                if choice == "1":
                    try:
                        addr, pk_hex, wif = create_wallet()
                        wallets[addr] = pk_hex
                        save_wallets(wallets)
                        print("Wallet created!")
                        print("Address (P2PKH):", addr)
                        print("Private Key (Hex):", pk_hex)
                        print("Private Key (WIF):", wif)
                        print("Please store the Private Key and WIF securely. Do not share them!")
                    except Exception as e:
                        print(f"Error creating wallet: {e}")
                elif choice == "2":
                    addr = input("Enter wallet address (P2PKH): ").strip()
                    try:
                        balance = utxo_set.get_balance(addr, blockchain.chain)
                        print(f"Balance of {addr}: {balance} PUK")
                    except Exception as e:
                        print(f"Error checking balance: {e}")
                elif choice == "3":
                    sender = input("Sender address (P2PKH): ").strip()
                    receiver = input("Receiver address (P2PKH): ").strip()
                    try:
                        amount = float(input("Amount: "))
                        fee = float(input("Transaction fee (PUK): "))
                        if sender not in wallets:
                            print("Private key for sender not stored locally!")
                            continue
                        total, inputs = utxo_set.select_utxos(sender, amount + fee, blockchain.chain)
                        if total < amount + fee:
                            print("Insufficient balance")
                            continue
                        outputs = [TXOutput(amount, receiver)]
                        if total > amount + fee:
                            outputs.append(TXOutput(total - amount - fee, sender))
                        tx = Transaction(inputs, outputs)
                        tx.sign(wallets[sender], utxo_set)
                        blockchain.add_transaction(tx, utxo_set)
                        p2p_node.broadcast_transaction(tx)
                        print(f"Transaction created and broadcast: {tx.txid}")
                    except Exception as e:
                        print(f"Transaction failed: {e}")
                elif choice == "4":
                    miner = input("Miner address (P2PKH): ").strip()
                    try:
                        block = blockchain.mine_pending(miner, utxo_set, oracle)
                        p2p_node.broadcast_block(block)
                        print(f"Mined block {block.index} | Hash: {block.hash} | Reward: {block.transactions[0].outputs[0].amount} PUK | TXs: {len(block.transactions)}")
                    except Exception as e:
                        print(f"Mining failed: {e}")
                elif choice == "5":
                    try:
                        for b in blockchain.chain:
                            print(f"Block {b.index} | Hash: {b.hash} | Prev Hash: {b.previous_hash} | Merkle Root: {b.merkle_root} | TXs: {len(b.transactions)}")
                            for tx in b.transactions:
                                print(f"  TX {tx.txid} {'(Coinbase)' if tx.coinbase_data else ''}")
                    except Exception as e:
                        print(f"Error viewing blockchain: {e}")
                elif choice == "6":
                    try:
                        blockchain.validate_chain()
                    except Exception as e:
                        print(f"Chain validation failed: {e}")
                elif choice == "7":
                    try:
                        addr, pk_hex, wif = create_wallet()
                        wallets[addr] = pk_hex
                        save_wallets(wallets)
                        print("Imported new wallet!")
                        print("Address (P2PKH):", addr)
                        print("Private Key (Hex):", pk_hex)
                        print("Private Key (WIF):", wif)
                        print("Please store the Private Key and WIF securely. Do not share them!")
                    except Exception as e:
                        print(f"Error importing new wallet: {e}")
                elif choice == "8":
                    try:
                        private_key_input = input("Enter private key (WIF or Hex): ").strip()
                        addr, pk_hex, wif = recover_wallet(private_key_input)
                        wallets[addr] = pk_hex
                        save_wallets(wallets)
                        print("Recovered wallet!")
                        print("Address (P2PKH):", addr)
                        print("Private Key (Hex):", pk_hex)
                        print("Private Key (WIF):", wif)
                        print("Please store the Private Key and WIF securely. Do not share them!")
                    except Exception as e:
                        print(f"Error recovering wallet: {e}")
                elif choice == "9":
                    peer_host = input("Peer host (e.g., localhost): ").strip()
                    peer_port = int(input("Peer port (e.g., 5001): "))
                    p2p_node.connect_to_peer(peer_host, peer_port)
                elif choice == "10":
                    p2p_node.stop()
                    break
                elif choice == "11":
                    try:
                        total_coins = blockchain.get_total_minted_coins()
                        print(f"Total Minted Coins: {total_coins} PUK")
                    except Exception as e:
                        print(f"Error checking total minted coins: {e}")
                elif choice == "12":
                    try:
                        source_amount = float(input("Amount to import from source chain (e.g., 1 BTC): "))
                        source_chain_id = input("Source chain ID (e.g., BTC, ETH): ").strip()
                        source_txid = input("Source transaction ID: ").strip()
                        receiver = input("Receiver address (P2PKH in PUK chain): ").strip()
                        private_key = input("Private key (Hex) for signing: ").strip()
                        exchange_request = ExchangeRequest(source_amount, source_chain_id, source_txid, receiver)
                        exchange_request.sign(private_key)
                        tx = blockchain.add_exchange_transaction(exchange_request, utxo_set)
                        p2p_node.broadcast_exchange(exchange_request)
                        print(f"Exchange request created and broadcast: {exchange_request.exchange_id}")
                        print(f"Converted {source_amount} {source_chain_id} to {exchange_request.amount} PUK at PUK price {oracle.get_puk_rate():.8f} USD")
                    except Exception as e:
                        print(f"Exchange failed: {e}")
                elif choice == "13":
                    try:
                        action = input("Action (set_burn_rate/add_liquidity/remove_liquidity/validate_pool): ").strip().lower()
                        if action == "set_burn_rate":
                            new_burn_rate = float(input("Enter new burn rate (e.g., 0.0001 for 0.01%): "))
                            oracle.pool.burn_rate = new_burn_rate
                            print(f"Burn rate updated to {new_burn_rate}")
                        elif action == "add_liquidity":
                            puk_amount = float(input("PUK amount to add: "))
                            usd_amount = float(input("USD amount to add: "))
                            oracle.pool.add_liquidity(puk_amount, usd_amount)
                        elif action == "remove_liquidity":
                            puk_amount = float(input("PUK amount to remove: "))
                            usd_amount = float(input("USD amount to remove: "))
                            oracle.pool.remove_liquidity(puk_amount, usd_amount)
                        elif action == "validate_pool":
                            oracle.pool.validate_pool()
                        else:
                            print("Invalid action")
                    except Exception as e:
                        print(f"Error managing liquidity pool: {e}")
                else:
                    print("Invalid option. Please enter a number between 1 and 13.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            p2p_node.stop()
            break
        except Exception as e:
            print(f"Error processing input: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()