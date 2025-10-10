# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
# Version 1.91: November 2024 fixed PT1 + PT2 parameter passing/unpacking
# Version 1.9: March 2024 added PT1+PT2, plus all the docstrings.
# Version 1.8; March 2023 added ZIPSH
# Version 1.7; September 2022 added PRDE
# Version 1.6; September 2021 added PRSH
# Version 1.5; 02 Jan 2021 -- was meant to be the final version before switch to BSE2.x, but that didn't happen :-)
# Version 1.4; 26 Oct 2020 -- change to Python 3.x
# Version 1.3; July 21st, 2018 (Python 2.x)
# Version 1.2; November 17th, 2012 (Python 2.x)
#
# Copyright (c) 2012-2024, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange's matching engine.
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel, with no fee/penalty imposed for doing so
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

import sys
import math
import random
import os
import time as chrono
import csv
import logging
from datetime import datetime

# LLM and belief graph imports
import google.generativeai as genai
from agents.belief_graph import BeliefGraph, MarketEvent, EventType
from TraderCustomAttributes import TraderCustomAttributes
import uuid
import json
import re
from dotenv import load_dotenv
from hm_trader import TraderLLM_HM

# Load environment variables from .env file
load_dotenv()

# a bunch of system constants (globals)
bse_sys_minprice = 1                    # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500                  # maximum price in the system, in cents/pennies
# ticksize should be a param of an exchange (so different exchanges can have different ticksizes)
ticksize = 1  # minimum change in price, in cents/pennies


# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:
    """
    An Order: this is used both for client-orders from exogenous customers to the robot traders acting as sales traders,
    and for the trader-orders (aka quotes) sent by the robot traders to the BSE exchange.
    In both use-cases, an order has a trader-i.d., a type (buy/sell), price, quantity, timestamp, and unique quote-i.d.
    """

    def __init__(self, tid, otype, price, qty, time, qid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.qid = qid  # quote i.d. (unique to each quote)

    def __str__(self):
        return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
               (self.tid, self.otype, self.price, self.qty, self.time, self.qid)


class OrderbookHalf:
    """
    OrderbookHalf is one side of the book: a list of bids or a list of asks, each sorted best-price-first,
    and with orders at the same price arranged by arrival time (oldest first) for time-priority processing.
    """

    def __init__(self, booktype, worstprice):
        """
        Create one side of the LOB
        :param booktype: specifies bid or ask side of the LOB.
        :param worstprice: the initial value of the worst price currently showing on the LOB.
        """
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None    # most extreme price quoted in this session
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        """
        anonymize a lob, strip out order details, format as a sorted list
        NB for asks, the sorting should be reversed
        :return: <nothing>
        """
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        """
        Take a list of orders and build a limit-order-book (lob) from it
        NB the exchange needs to know arrival times and trader-id associated with each order
        also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        :return: lob as a dictionary (i.e., unsorted)
        """
        lob_verbose = False
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        """
        Add order to the dictionary holding the list of orders for one side of the LOB.
        Either overwrites old order from this trader
            or dynamically creates new entry in the dictionary
            so there is a max of one order per trader per list
        checks whether length or order list has changed, to distinguish addition/overwrite
        :param order: the order to be added to the book
        :return: character-string indicating whether order-book was added to or overwritten.
        """

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        # add the order to the book
        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        """
        Delete order from the dictionary holding the orders for one half of the book.
        Assumes max of one order per trader per list.
        Checks that the Trader ID does actually exist in the dict before deletion.
        :param order: the order to be deleted.
        :return: <nothing>
        """
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        """
        When the best bid/ask has been hit/lifted, delete it from the book.
        :return: TraderID of the deleted order is return-value, as counterparty to the trade.
        """

        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


class Orderbook(OrderbookHalf):
    """ Orderbook for a single tradeable asset: list of bids and list of asks """

    def __init__(self):
        """Construct a new orderbook"""

        self.bids = OrderbookHalf('Bid', bse_sys_minprice)
        self.asks = OrderbookHalf('Ask', bse_sys_maxprice)
        self.tape = []
        self.tape_length = 10000    # max events on in-memory tape (older events can be written to tape_dump file)
        self.quote_id = 0           # unique ID code for each quote accepted onto the book
        self.lob_string = ''        # character-string linearization of public lob items with nonzero quantities


class Exchange(Orderbook):
    """  Exchange's matching engine and limit order book"""

    def add_order(self, order, vrbs):
        """
        add an order to the exchange -- either match with a counterparty order on LOB, or add to LOB.
        :param order: the order to be added to the LOB
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return: [order.qid, response] -- order.qid is the order's unique quote i.d., response is 'Overwrite'|'Addition'
        """
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        if vrbs:
            print('add_order QID=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, tape_file, vrbs):
        """
        Delete an order from the exchange.
        :param time: the current time.
        :param order: the order to be deleted from the LOB.
        :param tape_file: if not None, write details of the cancellation to the tape file.
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return: <nothing>
        """
        # delete a trader's quote/order from the exchange, update all internal records
        if vrbs:
            print('del_order QID=%d' % order.qid)
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            if tape_file is not None:
                tape_file.write('CAN, %f, %d, Bid, %d\n' % (time, order.qid, order.price))
            self.tape.append(cancel_record)
            # right-truncate the tape so that it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            if tape_file is not None:
                tape_file.write('CAN, %f, %d, Ask, %d\n' % (time, order.qid, order.price))
            self.tape.append(cancel_record)
            # right-truncate the tape so that it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order(self, time, order, tape_file, vrbs):
        """
        Process an order from a trader -- this is the BSE Matching Engine.
        :param time: the current time.
        :param order: the order to be processed.
        :param tape_file: if is not None then write details of transaction to tape_file
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return: transaction_record if the order results in a transaction, otherwise None.
        """
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None
        price = None
        [qid, response] = self.add_order(order, vrbs)  # add it to the order lists -- overwriting any previous order
        order.qid = qid
        if vrbs:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if vrbs:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if vrbs:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if vrbs:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if vrbs:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if vrbs:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            if vrbs:
                print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            if tape_file is not None:
                tape_file.write('TRD, %f, %d\n' % (time, price))
            self.tape.append(transaction_record)
            # right-truncate the tape so that it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

            return transaction_record
        else:
            return None

    def tape_dump(self, fname, fmode, tmode):
        """
        Currently tape_dump only writes a list of transactions (i.e., it ignores any cancellations)
        :param fname: filename to write to.
        :param fmode: file-open write/append mode.
        :param tmode: if set to 'wipe', wipes the tape clean after writing it to file.
        :return:
        """
        dumpfile = open(fname, fmode)
        dumpfile.write('Event Type, Time, Price\n')
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                dumpfile.write('Trd, %010.3f, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []

    def publish_lob(self, time, lob_file, vrbs):
        """
        Returns the public LOB data published by the exchange,
        i.e. the version of the LOB that's accessible to the traders.
        :param time: the current time.
        :param lob_file:
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return: the public LOB data.
        """
        public_data = dict()
        public_data['time'] = time

        # Build non-anonymous LOB with trader IDs: [[tid, price, qty], ...]
        bids_with_tids = []
        for price in sorted(self.bids.lob.keys(), reverse=True):
            orderlist = self.bids.lob[price][1]
            for order in orderlist:
                # order = [time, qty, tid, qid]
                bids_with_tids.append([order[2], price, order[1]])  # [tid, price, qty]

        asks_with_tids = []
        for price in sorted(self.asks.lob.keys()):
            orderlist = self.asks.lob[price][1]
            for order in orderlist:
                # order = [time, qty, tid, qid]
                asks_with_tids.append([order[2], price, order[1]])  # [tid, price, qty]

        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob': bids_with_tids}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': asks_with_tids}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape

        if lob_file is not None:
            # build a linear character-string summary of only those prices on LOB with nonzero quantities
            lobstring = 'Bid:,'
            n_bids = len(self.bids.lob_anon)
            if n_bids > 0:
                lobstring += '%d,' % n_bids
                for lobitem in self.bids.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            lobstring += 'Ask:,'
            n_asks = len(self.asks.lob_anon)
            if n_asks > 0:
                lobstring += '%d,' % n_asks
                for lobitem in self.asks.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            # is this different to the last lob_string?
            if lobstring != self.lob_string:
                # write it
                lob_file.write('%.3f, %s\n' % (time, lobstring))
                # remember it
                self.lob_string = lobstring

        if vrbs:
            vstr = 'publish_lob: t=%f' % time
            vstr += ' BID_lob=%s' % public_data['bids']['lob']
            # vstr += 'best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders)
            vstr += ' ASK_lob=%s' % public_data['asks']['lob']
            # vstr += 'qid=%d' % self.quote_id
            print(vstr)

        return public_data


# #################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:
    """The parent class for all types of robot trader in BSE"""

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initializes a generic trader with attributes common to all/most types of trader
        Some trader types (e.g. ZIP) then have additional specialised initialization steps
        :param ttype: the trader type
        :param tid: the trader I.D. (a non-negative integer)
        :param balance: how much money it has in the bank when it is created
        :param params: a set of parameter-values, for those trader-types that have parameters
        :param time: the time this trader was created
        """
        self.ttype = ttype          # what type / strategy this trader is
        self.tid = tid              # trader unique ID code
        self.balance = balance      # money in the bank
        self.params = params        # parameters/extras associated with this trader-type or individual trader.
        self.blotter = []           # record of trades executed
        self.blotter_length = 100   # maximum length of blotter
        self.orders = []            # customer orders currently being worked (fixed at len=1 in BSE1.x)
        self.n_quotes = 0           # number of quotes live on LOB
        self.birthtime = time       # used when calculating age of a trader/strategy
        self.profitpertime = 0      # profit per unit time
        self.profit_mintime = 60    # minimum duration in seconds for calculating profitpertime
        self.n_trades = 0           # how many trades has this trader done?
        self.lastquote = None       # record of what its last quote was

    def __str__(self):
        """ return a character-string that summarises a trader """
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, vrbs):
        """
        What a trader calls when it receives a new customer order/assignment
        :param order: the customer order/assignment to be added
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return response: string to indicate whether the trader needs to cancel its current order on the LOB
        """
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if vrbs:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        """What a trader calls when it wants to delete an existing customer order/assignment """
        if order is None:
            pass    # this line is purely to stop PyCharm from warning about order being an unused parameter
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        self.orders = []

    def profitpertime_update(self, time, birthtime, totalprofit):
        """
        Calculates the trader's profit per unit time, but only if it has been alive longer than profit_mintime
        This is to avoid situations where a trader is created and then immediately makes a profit and
        hence the profit per unit time is a sky-high value, because the time_alive divisor is close to zero.
        :param time: the current time.
        :param birthtime: the time when the trader was created.
        :param totalprofit: the trader's current total accumulated profit.
        :return: profit per second.
        """
        time_alive = (time - birthtime)
        if time_alive >= self.profit_mintime:
            profitpertime = totalprofit / time_alive
        else:
            # if it's not been alive long enough, divide it by mintime instead of actual time
            profitpertime = totalprofit / self.profit_mintime
        return profitpertime

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's individual records of transactions, profit/loss etc.
        :param trade: details of the transaction that took place.
        :param order: details of the customer order that led to the transaction.
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :param time: the current time.
        :return: <nothing>
        """
        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)
        
        # Add transaction prints for all trader types
        if self.orders[0].otype == "Bid":
            # Bought something
            print(f"💰 {self.ttype} {self.tid} BOUGHT at ${transactionprice} | Balance: ${self.balance:.0f}")
        else:
            # Sold something
            if profit > 0:
                print(f"🟢 {self.ttype} {self.tid} SOLD at ${transactionprice} | Profit: ${profit:.0f} | Balance: ${self.balance:.0f}")
            elif profit == 0:
                print(f"🟡 {self.ttype} {self.tid} SOLD at ${transactionprice} | Break-even | Balance: ${self.balance:.0f}")
            else:
                print(f"🔴 {self.ttype} {self.tid} SOLD at ${transactionprice} | Loss: ${abs(profit):.0f} | Balance: ${self.balance:.0f}")
        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('FAIL: negative profit')

        if vrbs:
            print('%s profit=%d balance=%d profit/time=%s' % (outstr, profit, self.balance, str(self.profitpertime)))
        self.del_order(order)  # delete the order

        # if the trader has multiple strategies (e.g. PRSH/PRDE/ZIPSH/ZIPDE) then there is more work to do...
        if hasattr(self, 'strats') and hasattr(self, 'active_strat'):
            if self.strats is not None:
                self.strats[self.active_strat]['profit'] += profit
                totalprofit = self.strats[self.active_strat]['profit']
                birthtime = self.strats[self.active_strat]['start_t']
                self.strats[self.active_strat]['pps'] = self.profitpertime_update(time, birthtime, totalprofit)

    def respond(self, time, lob, trade, vrbs):
        """
        Specify how a trader responds to events in the market.
        For Trader superclass, this is minimal action, but expect it to be overloaded by specific trading strategies.
        :param time:
        :param lob:
        :param trade:
        :param vrbs: verbosity: if True, print a running commentary; if False, stay silent.
        :return:
        """

        # any trader subclass with custom respond() must include this update of profitpertime
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        return None


class TraderGiveaway(Trader):
    """
    Trader subclass Giveaway (GVWY): even dumber than a ZI-U: just give the deal away (but never make a loss)
    """

    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        :param time: the current time.
        :param countdown: how much time before market closes (not used by GVWY).
        :param lob: the current state of the LOB.
        :return: a new order from this trader.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order


class TraderZIC(Trader):
    """
    Trader subclass ZI-C: after Gode & Sunder 1993
    """

    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        :param time: the current time.
        :param countdown: how much time before market closes (not used by ZIC).
        :param lob: the current state of the LOB.
        :return: a new order from this trader.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                quoteprice = random.randint(int(minprice), int(limit))
            else:
                quoteprice = random.randint(int(limit), int(maxprice))
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order
        return order


class TraderShaver(Trader):
    """
    Trader subclass Shaver: shaves a penny off the best price;
    but if there is no best price, creates "stub quote" at system max/min
    """

    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        :param time: the current time.
        :param countdown: how much time before market close (not used by SHVR).
        :param lob: the current state of the LOB.
        :return: a new order from this trader.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1:
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + 1
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - 1
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


class TraderSniper(Trader):
    """
    Trader subclass Sniper (SNPR), inspired by Kaplan's Sniper, BSE version is based on Shaver,
    "lurks" until time remaining < threshold% of the trading session
    then gets increasing aggressive, increasing "shave thickness" as time runs out
    """

    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        :param time: the current time.
        :param countdown: how much time before market closes.
        :param lob: the current state of the LOB.
        :return: a new order from this trader.
        """
        lurk_threshold = 0.2
        shavegrowthrate = 3
        shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + shave
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - shave
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


class TraderPRZI(Trader):
    """
    Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader -- pronounced "prezzie"
    but with added adaptive strategies, currently either...
       ++  a k-point Stochastic Hill-Climber (SHC) hence PRZI-SHC,
           PRZI-SHC pronounced "prezzy-shuck". Ticker symbol PRSH pronounced "purrsh";
    or
       ++ a simple differential evolution (DE) optimizer with pop_size=k, hence PRZE-DE or PRDE ('purdy")

    when optimizer == None then it implements plain-vanilla non-adaptive PRZI, with a fixed strategy-value.
    """

    @staticmethod
    def strat_csv_str(strat):
        """
        Return trader's strategy as a csv-format string
        (trivial in PRZI, but other traders with more complex strategies need this).
        :param strat: the strategy specification (for PRZI, a real number in [-1.0,+1.0]
        :return: the strategy as a scv-format string
        """
        csv_str = 's=,%+5.3f, ' % strat
        return csv_str

    def mutate_strat(self, s, mode):
        """
        How to mutate the PRZI strategy values when evolving / hill-climbing
        :param s: the strategy to be mutated
        :param mode:    'gauss'=> mutation is a draw from a zero-mean Gaussian;
                        'uniform_whole_range" => mutation is a draw from uniform distbn over [-1.0,+1.0].
                        'uniform_bounded_range" => mutation is a draw from a bounded unifrom distbn.
        :return: the mutated strategy value
        """
        s_min = self.strat_range_min
        s_max = self.strat_range_max
        if mode == 'gauss':
            sdev = 0.05
            newstrat = s
            while newstrat == s:
                newstrat = s + random.gauss(0.0, sdev)
                # truncate to keep within range
                newstrat = max(-1.0, min(1.0, newstrat))
        elif mode == 'uniform_whole_range':
            # draw uniformly from whole range
            newstrat = random.uniform(-1.0, +1.0)
        elif mode == 'uniform_bounded_range':
            # draw uniformly from bounded range
            newstrat = random.uniform(s_min, s_max)
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat

    def strat_str(self):
        """
        Pretty-print a string summarising this trader's strategy/strategies
        :return: the string
        """
        string = '%s: %s active_strat=[%d]:\n' % (self.tid, self.ttype, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%+f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string

    def __init__(self, ttype, tid, balance, params, time):
        """
        Construct a PRZI trader
        :param ttype: the ticker-symbol for the type of trader (its strategy)
        :param tid: the trader id
        :param balance: the trader's bank balance
        :param params: if params == "landscape-mapper" then it generates data for mapping the fitness landscape
        :param time: the current time.
        """

        vrbs = True

        Trader.__init__(self, ttype, tid, balance, params, time)

        # unpack the params
        # for all three of PRZI, PRSH, and PRDE params can include strat_min and strat_max
        # for PRSH and PRDE params should include values for optimizer and k
        # if no params specified then defaults to PRZI with strat values in [-1.0,+1.0]

        # default parameter values
        k = 1
        optimizer = None    # no optimizer => plain non-adaptive PRZI
        s_min = -1.0
        s_max = +1.0

        # did call provide different params?
        if type(params) is dict:
            if 'k' in params:
                k = params['k']
            if 'optimizer' in params:
                optimizer = params['optimizer']
            s_min = params['strat_min']
            s_max = params['strat_max']

        self.optmzr = optimizer     # this determines whether it's PRZI, PRSH, or PRDE
        self.k = k                  # number of sampling points (cf number of arms on a multi-armed-bandit, or pop-size)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.strat_wait_time = 7200     # how many secs do we give any one strat before switching?
        self.strat_range_min = s_min    # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = s_max    # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None        # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time   # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.0 * random.random()    # min profit-per-sec difference between strategies that counts
        self.strats = []            # strategies awaiting initialization
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1, 10))  # multiplier coefficient when estimating p_max
        self.mapper_outfile = None
        # differential evolution parameters all in one dictionary
        self.diffevol = {'de_state': 'active_s0',          # initial state: strategy 0 is active (being evaluated)
                         's0_index': self.active_strat,    # s0 starts out as active strat
                         'snew_index': self.k,             # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,            # assigned later
                         'F': 0.8                          # differential weight -- usually between 0 and 2
                         }

        start_t = time
        profit = 0.0
        profit_per_second = 0
        lut_bid = None
        lut_ask = None

        for s in range(self.k + 1):
            # initialise each of the strategies in sequence:
            # for PRZI: only one strategy is needed
            # for PRSH, one random initial strategy, then k-1 mutants of that initial strategy
            # for PRDE, use draws from uniform distbn over whole range and a (k+1)th strategy is needed to hold s_new
            strategy = None
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                if self.optmzr == 'PRSH':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')     # mutant of strats[0]
                elif self.optmzr == 'PRDE':
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_bounded_range')
                else:
                    # plain PRZI -- do nothing
                    pass
            # add to the list of strategies
            if s == self.active_strat:
                active_flag = True
            else:
                active_flag = False
            self.strats.append({'stratval': strategy, 'start_t': start_t, 'active': active_flag,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
            if self.optmzr is None:
                # PRZI -- so we stop after one iteration
                break
            elif self.optmzr == 'PRSH' and s == self.k - 1:
                # PRSH -- doesn't need the (k+1)th strategy
                break

        if self.params == 'landscape-mapper':
            # replace seed+mutants set of strats with regularly-spaced strategy values over the whole range
            self.strats = []
            strategy_delta = 0.01
            strategy = -1.0
            k = 0
            self.strats = []

            while strategy <= +1.0:
                self.strats.append({'stratval': strategy, 'start_t': start_t, 'active': False,
                                    'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
                k += 1
                strategy += strategy_delta
            self.mapper_outfile = open('landscape_map.csv', 'w')
            self.k = k
            self.strat_eval_time = self.k * self.strat_wait_time

        if vrbs:
            print("%s\n" % self.strat_str())

    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        :param time: the current time.
        :param countdown: how much time before market close (not used by GVWY).
        :param lob: the current state of the LOB.
        :return: a new order from this trader.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        def shvr_price(order_type, lim, pub_lob):
            """
            Return value is what price a SHVR would quote in these circumstances
            :param order_type: is the order bid or ask?
            :param lim: limit price on the order.
            :param pub_lob: the current state of the published LOB.
            :return: The price a SHVR would quote given this LOB and limit-price.
            """

            if order_type == 'Bid':
                if pub_lob['bids']['n'] > 0:
                    shvr_p = pub_lob['bids']['best'] + ticksize   # BSE ticksize is global var
                    if shvr_p > lim:
                        shvr_p = lim
                else:
                    shvr_p = pub_lob['bids']['worst']
            else:
                if pub_lob['asks']['n'] > 0:
                    shvr_p = pub_lob['asks']['best'] - ticksize   # BSE ticksize is global var
                    if shvr_p < lim:
                        shvr_p = lim
                else:
                    shvr_p = pub_lob['asks']['worst']

            # print('shvr_p=%f; ' % shvr_p)
            return shvr_p

        def calc_cdf_lut(strategy, t0, m, dirn, pmin, pmax):
            """
            calculate cumulative distribution function (CDF) look-up table (LUT)
            :param strategy: strategy-value in [-1,+1]
            :param t0: constant used in the threshold function
            :param m: constant used in the threshold function
            :param dirn: direction: 'buy' or 'sell'
            :param pmin: lower bound on discrete-valued price-range
            :param pmax: upper bound on discrete-valued price-range
            :return: {'strat': strategy, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}
            """

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.000001  # used to catch DIV0 errors
            lut_vrbs = False

            if (strategy > 1.0) or (strategy < -1.0):
                # out of range
                sys.exit('PRSH FAIL: strategy=%f out of range\n' % strategy)

            if (dirn != 'buy') and (dirn != 'sell'):
                # out of range
                sys.exit('PRSH FAIL: bad dirn=%s\n' % dirn)

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax %f < pmin %f \n' % (pmax, pmin))

            if lut_vrbs:
                print('PRSH calc_cdf_lut: strategy=%f dirn=%d pmin=%d pmax=%d\n' % (strategy, dirn, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the limit-price with probability 1

                if dirn == 'buy':
                    cdf = [{'price': pmax, 'cum_prob': 1.0}]
                else:   # must be a sell
                    cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if lut_vrbs:
                    print('\n\ncdf:', cdf)

                return {'strat': strategy, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strategy + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                # normalize the price to proportion of its range
                p_r = (p - pmin) / p_range  # p_r in [0.0, 1.0]
                if strategy == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strategy > 0:
                    if dirn == 'buy':
                        cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                    else:   # dirn == 'sell'
                        cal_p = (math.exp(c * (1 - p_r)) - 1.0) / e2cm1
                else:   # self.strat < 0
                    if dirn == 'buy':
                        cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                    else:   # dirn == 'sell'
                        cal_p = 1.0 - ((math.exp(c * (1 - p_r)) - 1.0) / e2cm1)

                if cal_p < 0:
                    cal_p = 0   # just in case

                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if lut_vrbs:
                print('\n\ncdf:', cdf)

            return {'strat': strategy, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        vrbs = False

        if vrbs:
            print('t=%.1f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                pass

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible as defined by exchange

            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax    # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)     # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']         # so use that as my new estimate of highest
                    self.pmax = maxprice

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            if otype == 'Bid':

                p_max = int(limit)
                if strat > 0.0:
                    p_min = minprice
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']

                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or (lut_bid['pmin'] != p_min) or (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if vrbs:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = \
                        calc_cdf_lut(strat, self.theta0, self.m, 'buy', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:   # otype == 'Ask'

                p_min = int(limit)
                if strat > 0.0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * maxprice))
                    if p_max < p_min:
                        # this should never happen, but just in case it does...
                        p_max = p_min

                lut_ask = self.strats[self.active_strat]['lut_ask']

                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if vrbs:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = \
                        calc_cdf_lut(strat, self.theta0, self.m, 'sell', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_ask']

            vrbs = False
            if vrbs:
                print('PRZI strat=%f LUT=%s \n \n' % (strat, lut))
                # for debugging: print a table of lut: price and cum_prob, with the discrete derivative (gives PMF).
                last_cprob = 0.0
                for lut_entry in lut['cdf_lut']:
                    cprob = lut_entry['cum_prob']
                    print('%d, %f, %f' % (lut_entry['price'], cprob - last_cprob, cprob))
                    last_cprob = cprob
                print('\n')
                
                # print ('[LUT print suppressed]')
            
            # do inverse lookup on the LUT to find the price
            quoteprice = None
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's individual records of transactions, profit/loss etc.
        :param trade: details of the transaction that took place
        :param order: details of the customer order that led to the transaction
        :param vrbs: if True then print a running commentary of what's going on
        :param time: the current time
        :return: (nothing)
        """

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]      # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if vrbs:
            print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        self.strats[self.active_strat]['profit'] += profit
        time_alive = time - self.strats[self.active_strat]['start_t']
        if time_alive > 0:
            profit_per_second = self.strats[self.active_strat]['profit'] / time_alive
            self.strats[self.active_strat]['pps'] = profit_per_second
        else:
            # if it trades at the instant it is born then it would have infinite profit-per-second, which is insane
            # to keep things sensible when time_alive == 0 we say the profit per second is whatever the actual profit is
            self.strats[self.active_strat]['pps'] = profit

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to the current state of the LOB.
        For strategy-optimizers PRSH and PRDE, this can involve switching stratregy, and/or generating new strategies.
        :param time: the current time.
        :param lob: the current state of the LOB.
        :param trade: details of most recent trade, if any.
        :param vrbs: if True then print messages explaining what is going on.
        :return:
        """
        # "PRSH" is a very basic form of stochastic hill-climber (SHC) that's v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other k-1 strats
        # then all counters are reset, and this is repeated indefinitely
        #
        # "PRDE" uses a basic form of Differential Evolution. This maintains a population of at least four strats
        # iterates indefinitely on:
        #       shuffle the set of strats;
        #       name the first four strats s0 to s3;
        #       create new_strat=s1+f*(s2-s3);
        #       evaluate fitness of s0 and new_strat;
        #       if (new_strat fitter than s0) then new_strat replaces s0.
        #
        # todo: add in other optimizer algorithms that are cleverer than these
        #  e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        def strat_activate(t, s_index):
            """
            Activate a specified strategy
            :param t: the current time
            :param s_index: the index of the strategy in the list of strategies
            :return: <nothing>
            """
            # print('t=%f Strat_activate, index=%d, active=%s' % (t, s_index, self.strats[s_index]['active'] ))
            self.strats[s_index]['start_t'] = t
            self.strats[s_index]['active'] = True
            self.strats[s_index]['profit'] = 0.0
            self.strats[s_index]['pps'] = 0.0

        vrbs = False

        # first update each active strategy's profit-per-second (pps) value -- this is the "fitness" of each strategy
        for s in self.strats:
            # debugging check: make profit be directly proportional to strategy, no noise
            # s['profit'] = 100 * abs(s['stratval'])
            # update pps
            active_flag = s['active']
            if active_flag:
                s['pps'] = self.profitpertime_update(time, s['start_t'], s['profit'])

        if self.optmzr == 'PRSH':

            if vrbs:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                pass

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy
                self.strats[s]['active'] = False

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.strats[new_strat]['active'] = True
                self.last_strat_change_time = time

                if vrbs:
                    swt = self.strat_wait_time
                    print('t=%.3f (%.2fdays), %s PRSHrespond: strat[%d] elpsd=%.3f; wait_t=%.3f, pps=%f, new strat=%d' %
                          (time, time/86400, self.tid, s, time_elapsed, swt, self.strats[s]['pps'], new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            # assume that all strats have had long enough, and search for evidence to the contrary
            all_old_enough = True
            for s in self.strats:
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if vrbs:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

                if self.params == 'landscape-mapper':
                    for s in self.strats:
                        self.mapper_outfile.write('time, %f, strat, %f, pps, %f\n' %
                                                  (time, s['stratval'], s['pps']))
                    self.mapper_outfile.flush()
                    sys.exit()

                else:
                    # if the difference between the top two strats is too close to call then flip a coin
                    # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                    best_strat = 0
                    prof_diff = strats_sorted[0]['pps'] - strats_sorted[1]['pps']
                    if abs(prof_diff) < self.profit_epsilon:
                        # they're too close to call, so just flip a coin
                        best_strat = random.randint(0, 1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = strats_sorted[0]
                        strats_sorted[0] = strats_sorted[1]
                        strats_sorted[1] = tmp_strat

                    # the sorted list of strats replaces the existing list
                    self.strats = strats_sorted

                    # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate

                    # now replicate and mutate the elite into all the other strats
                    for s in range(1, self.k):    # note range index starts at one not zero (elite is at [0])
                        self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], 'gauss')
                        self.strats[s]['start_t'] = time
                        self.strats[s]['profit'] = 0.0
                        self.strats[s]['pps'] = 0.0
                    # and then update (wipe) records for the elite
                    self.strats[0]['start_t'] = time
                    self.strats[0]['profit'] = 0.0
                    self.strats[0]['pps'] = 0.0
                    self.active_strat = 0

                if vrbs:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

        elif self.optmzr == 'PRDE':
            # simple differential evolution

            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    self.strats[self.active_strat]['active'] = False
                    # we've evaluated s0, so now we need to evaluate s_new
                    self.active_strat = self.diffevol['snew_index']
                    strat_activate(time, self.active_strat)

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # now we've evaluated s_0 and s_new, so we can do DE adaptive step
                    if vrbs:
                        print('PRDE trader %s' % self.tid)
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']

                    if verbose:
                        print('DiffEvol: t=%.1f, i_0=%d, i0fit=%f, i_new=%d, i_new_fit=%f' %
                              (time, i_0, fit_0, i_new, fit_new))

                    if fit_new >= fit_0:
                        # new strat did better than old strat0, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))    # create sequential list of strategy-numbers
                    random.shuffle(stratlist)             # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    if verbose:
                        print('DiffEvol: t=%.1f, s0=%d, s1=%d, (s=%+f), s2=%d, (s=%+f), s3=%d, (s=%+f), sNew=%+f' %
                              (time, self.diffevol['s0_index'],
                               s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    strat_sum = 0.0
                    for s in range(self.k):
                        strat_sum += self.strats[s]['stratval']
                    strat_mean = strat_sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    if vrbs:
                        print('t=,%.1f, MeanStrat=, %+f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one strategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        if verbose:
                            print('Converged pop: set strategy %d to %+f' %
                                  (randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    strat_activate(time, self.active_strat)

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        elif self.optmzr is None:
            # this is PRZI -- nonadaptive, no optimizer, nothing to change here.
            pass

        else:
            sys.exit('FAIL: bad value for self.optmzr')


class TraderZIP(Trader):
    """
    The Zero-Intelligence-Plus (ZIP) adaptive trading strategy of Cliff (1997).
    The code here implements the original ZIP, and also the strategy-optimizing variuants ZIPSH and ZIPDE.
    """

    # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    # NB this implementation keeps separate margin values for buying & selling,
    #    so a single trader can both buy AND sell
    #    -- in the original, traders were either buyers OR sellers

    @staticmethod
    def strat_csv_str(strat):
        """
        Take a ZIP strategy vector and return it as a csv-format string.
        :param strat: the vector of values for the ZIP trader's strategy
        :return: the csv-format string.
        """
        if strat is None:
            csv_str = 'None, '
        else:
            csv_str = 'mBuy=,%+5.3f, mSel=,%+5.3f, b=,%5.3f, m=,%5.3f, ca=,%6.4f, cr=,%6.4f, ' % \
                      (strat['m_buy'], strat['m_sell'], strat['beta'], strat['momntm'], strat['ca'], strat['cr'])
        return csv_str

    @staticmethod
    def mutate_strat(s, mode):
        """
        How to mutate the strategy values when evolving / hill-climbing
        :param s: the strategy to be mutated.
        :param mode: specify Gaussian or some other form of distribution for the mutation delta (currently only Gauss).
        :return: the mutated strategy.
        """

        def gauss_mutate_clip(value, sdev, range_min, range_max):
            """
            Mutation of strategy-value by injection of zero-mean Gaussian noise, followed by clipping to keep in range.
            :param value: the value to be mutated.
            :param sdev: the standard deviation on the Gaussian noise.
            :param range_min: lower bound on the range.
            :param range_max: upper bound opn the range.
            :return: the mutated value.
            """
            mut_val = value
            while mut_val == value:
                mut_val = value + random.gauss(0.0, sdev)
                if mut_val > range_max:
                    mut_val = range_max
                elif mut_val < range_min:
                    mut_val = range_min
            return mut_val

        # mutate each element of a ZIP strategy independently
        # and clip each to remain within bounds
        if mode == 'gauss':
            big_sdev = 0.025
            small_sdev = 0.0025
            margin_buy = gauss_mutate_clip(s['m_buy'], big_sdev, -1.0, 0)
            margin_sell = gauss_mutate_clip(s['m_sell'], big_sdev, 0.0, 1.0)
            beta = gauss_mutate_clip(s['beta'], big_sdev, 0.0, 1.0)
            momntm = gauss_mutate_clip(s['momntm'], big_sdev, 0.0, 1.0)
            ca = gauss_mutate_clip(s['ca'], small_sdev, 0.0, 1.0)
            cr = gauss_mutate_clip(s['cr'], small_sdev, 0.0, 1.0)
            new_strat = {'m_buy': margin_buy, 'm_sell': margin_sell, 'beta': beta, 'momntm': momntm, 'ca': ca, 'cr': cr}
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return new_strat

    def __init__(self, ttype, tid, balance, params, time):
        """
        Create a ZIP/ZIPSH/ZIPDE trader.
        :param ttype: the string identifying the trader-type (what strategy is this).
        :param tid: the trader i.d. string.
        :param balance: the starting bank balance for this trader.
        :param params: any additional parameters.
        :param time: the current time.
        """

        Trader.__init__(self, ttype, tid, balance, params, time)

        # this set of one-liner functions named init_*() are just to make the init params obvious for ease of editing
        # for ZIP, a strategy is specified as a 6-tuple: (margin_buy, margin_sell, beta, momntm, ca, cr)
        # the 'default' values mentioned in comments below come from Cliff 1997 -- good ranges for most situations

        def init_beta():
            """in Cliff 1997 the initial beta values are U(0.1, 0.5)"""
            return random.uniform(0.1, 0.5)

        def init_momntm():
            """in Cliff 1997 the initial momentum values are U(0.0, 0.1)"""
            return random.uniform(0.0, 0.1)

        def init_ca():
            # in Cliff 1997 c_a was a system constant, the same for all traders, set to 0.05
            # here we take the liberty of introducing some variation
            return random.uniform(0.01, 0.05)

        def init_cr():
            # in Cliff 1997 c_r was a system constant, the same for all traders, set to 0.05
            # here we take the liberty of introducing some variation
            return random.uniform(0.01, 0.05)

        def init_margin():
            # in Cliff 1997 the initial margin values are U(0.05, 0.35)
            return random.uniform(0.05, 0.35)

        def init_stratwaittime():
            # not in Cliff 1997: use whatever limits you think best.
            return 7200 + random.randint(0, 3600)

        # unpack the params
        # for ZIPSH and ZIPDE params should include values for optimizer and k
        # if no params specified then defaults to ZIP with strat values as in Cliff1997

        # default parameter values
        k = 1
        optimizer = None    # no optimizer => plain non-optimizing ZIP
        logging = False

        # did call provide different params?
        if type(params) is dict:
            if 'k' in params:
                k = params['k']
            if 'optimizer' in params:
                optimizer = params['optimizer']
            self.logfile = None
            if 'logfile' in params:
                logging = True
                logfilename = params['logfile'] + '_' + tid + '_log.csv'
                self.logfile = open(logfilename, 'w')

        # the following set of variables are needed for original ZIP *and* for its optimizing extensions e.g. ZIPSH
        self.logging = logging
        self.willing = 1
        self.able = 1
        self.job = None             # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False         # gets switched to True while actively working an order
        self.prev_change = 0        # this was called last_d in Cliff'97
        self.beta = init_beta()
        self.momntm = init_momntm()
        self.ca = init_ca()         # self.ca & self.cr were hard-coded in '97 but parameterised later
        self.cr = init_cr()
        self.margin = None          # this was called profit in Cliff'97
        self.margin_buy = -1.0 * init_margin()
        self.margin_sell = init_margin()
        self.price = None
        self.limit = None
        self.prev_best_bid_p = None     # best bid price on LOB on previous update
        self.prev_best_bid_q = None     # best bid quantity on LOB on previous update
        self.prev_best_ask_p = None     # best ask price on LOB on previous update
        self.prev_best_ask_q = None     # best ask quantity on LOB on previous update

        # the following set of variables are needed only by ZIP with added hyperparameter optimization (e.g. ZIPSH)
        self.k = k                  # how many strategies evaluated at any one time?
        self.optmzr = optimizer     # what form of strategy-optimizer we're using
        self.strats = None          # the list of strategies, each of which is a dictionary
        self.strat_wait_time = init_stratwaittime()     # how many secs do we give any one strat before switching?
        self.strat_eval_time = self.k * self.strat_wait_time  # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.profit_epsilon = 0.0 * random.random()     # min profit-per-sec difference between strategies that counts

        if self.optmzr is not None and k > 1:
            # we're doing some form of k-armed strategy-optimization with multiple strategies
            self.strats = []
            # strats[0] is whatever we've just assigned, and is the active strategy
            strategy = {'m_buy': self.margin_buy, 'm_sell': self.margin_sell, 'beta': self.beta,
                        'momntm': self.momntm, 'ca': self.ca, 'cr': self.cr}
            self.strats.append({'stratvec': strategy, 'start_t': time, 'active': True,
                                'profit': 0, 'pps': 0, 'evaluated': False})

            # rest of *initial* strategy set is generated from same distributions, but these are all inactive
            for s in range(1, k):
                strategy = {'m_buy': -1.0 * init_margin(), 'm_sell': init_margin(), 'beta': init_beta(),
                            'momntm': init_momntm(), 'ca': init_ca(), 'cr': init_cr()}
                self.strats.append({'stratvec': strategy, 'start_t': time, 'active': False,
                                    'profit': 0, 'pps': 0, 'evaluated': False})

        if self.logging:
            self.logfile.write('ZIP, Tid, %s, ttype, %s, optmzr, %s, strat_wait_time, %f, n_strats=%d:\n' %
                               (self.tid, self.ttype, self.optmzr, self.strat_wait_time, self.k))
            for s in self.strats:
                self.logfile.write(str(s)+'\n')

    def getorder(self, time, countdown, lob):
        """
        Create the next order for this trader
        :param time: the current time
        :param countdown: time remaining until market closes (not used in ZIP)
        :param lob: the current state of the LOB
        :return: this trader's next order.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quoteprice = int(self.limit * (1 + self.margin))

            lastprice = -1  # dummy value for if there is no lastprice
            if self.lastquote is not None:
                lastprice = self.lastquote.price

            self.price = quoteprice
            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order

            if self.logging and order.price != lastprice:
                self.logfile.write('%f, Order:, %s\n' % (time, str(order)))
        return order

    def respond(self, time, lob, trade, vrbs):
        """
        Update ZIP profit margin on basis of what happened in market.
        For ZIPSH and ZIPDE, also maybe switch strategy and/or generate new strategies to evaluate.
        :param time: the current time.
        :param lob: the current state of the LOB.
        :param trade: details of most recent trade, if any.
        :param vrbs: if True then print a running commentary of what is going on.
        :return: snapshot: if Ture, then the caller of respond() should print the next frame of system snapshot data.
        """
        # ZIP trader responds to market events, altering its margin
        # does this whether it currently has an order to work or not

        def target_up(price):
            """ Generate a higher target price by randomly perturbing given price"""
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))
            # #                        print('TargetUp: %d %d\n' % (price,target))
            return target

        def target_down(price):
            """ Generate a lower target price by randomly perturbing given price"""
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))
            # #                        print('TargetDn: %d %d\n' % (price,target))
            return target

        def willing_to_trade(price):
            """ Am I willing to trade at this price?"""
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            """
            ZIP profit-margin update on basis of target price -- updates self.margin.
            :param price: the target price.
            :return: <nothing>
            """
            oldprice = self.price
            diff = price - oldprice
            change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
            self.prev_change = change
            newmargin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if newmargin < 0.0:
                    self.margin_buy = newmargin
                    self.margin = newmargin
            else:
                if newmargin > 0.0:
                    self.margin_sell = newmargin
                    self.margin = newmargin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        def load_strat(stratvec, birthtime):
            """
            Copy the strategy vector into the ZIP trader's params and timestamp it.
            :param stratvec: the strategy vector.
            :param birthtime: the timestamp.
            :return: <nothing>
            """
            self.margin_buy = stratvec['m_buy']
            self.margin_sell = stratvec['m_sell']
            self.beta = stratvec['beta']
            self.momntm = stratvec['momntm']
            self.ca = stratvec['ca']
            self.cr = stratvec['cr']
            # bookkeeping
            self.n_trades = 0
            self.birthtime = birthtime
            self.balance = 0
            self.profitpertime = 0

        def strat_activate(t, s_index):
            """
            Activate a specified strategy-vector.
            :param t: the current time.
            :param s_index: the index of the strategy to be activated.
            :return: <nothing>
            """
            # print('t=%f Strat_activate, index=%d, active=%s' % (t, s_index, self.strats[s_index]['active'] ))
            self.strats[s_index]['start_t'] = t
            self.strats[s_index]['active'] = True
            self.strats[s_index]['profit'] = 0.0
            self.strats[s_index]['pps'] = 0.0
            self.strats[s_index]['evaluated'] = False

        # snapshot says whether the caller of respond() should print next frame of system snapshot data
        snapshot = False

        if self.optmzr == 'ZIPSH':

            # ZIP with simple-stochastic-hillclimber optimization of strategy (hyperparameter values)

            # NB this *cycles* through the available strats in sequence (i.e., it doesn't shuffle them)

            # first update the pps for each active strategy
            for s in self.strats:
                # update pps
                active_flag = s['active']
                if active_flag:
                    s['pps'] = self.profitpertime_update(time, s['start_t'], s['profit'])

            # have we evaluated all the strategies?
            # (could instead just compare active_strat to k, but checking them all in sequence is arguably clearer)
            # assume that all strats have been evaluated, and search for evidence to the contrary
            all_evaluated = True
            for s in self.strats:
                if s['evaluated'] is False:
                    all_evaluated = False
                    break

            if all_evaluated:
                # time to generate a new set/population of k candidate strategies
                # NB when the final strategy in the trader's set/popln is evaluated, the set is then sorted into
                # descending order of profitability, so when we get to here we know that strats[0] is elite

                if vrbs and self.tid == 'S00':
                    print('t=%.3f, ZIPSH %s: strat_eval_time=%.3f,' % (time, self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('%s, start_t=%f, $=%f, pps=%f' %
                              (self.strat_csv_str(s['stratvec']), s['start_t'], s['profit'], s['pps']))

                # if the difference between the top two strats is too close to call then flip a coin
                # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                best_strat = 0
                prof_diff = self.strats[0]['pps'] - self.strats[1]['pps']
                if abs(prof_diff) < self.profit_epsilon:
                    # they're too close to call, so just flip a coin
                    best_strat = random.randint(0, 1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = self.strats[0]
                        self.strats[0] = self.strats[1]
                        self.strats[1] = tmp_strat

                # at this stage, strats[0] is our newly-chosen elite-strat, about to replicate & mutate

                # now replicate and mutate the elite into all the other strats
                for s in range(1, self.k):  # note range index starts at one not zero (elite is at [0])
                    self.strats[s]['stratvec'] = self.mutate_strat(self.strats[0]['stratvec'], 'gauss')
                    strat_activate(time, s)

                # and then update (wipe) records for the elite
                strat_activate(time, 0)

                # load the elite into the ZIP trader params
                load_strat(self.strats[0]['stratvec'], time)

                self.active_strat = 0

                if vrbs and self.tid == 'S00':
                    print('%s: strat_eval_time=%f, best_strat=%d, MUTATED:' %
                          (self.tid, self.strat_eval_time, best_strat))
                    for s in self.strats:
                        print('%s start_t=%.3f, lifetime=%.3f, $=%.3f, pps=%f' %
                              (self.strat_csv_str(s['stratvec']), s['start_t'], time - s['start_t'], s['profit'],
                               s['pps']))

            else:
                # we're still evaluating

                s = self.active_strat
                time_elapsed = time - self.strats[s]['start_t']
                if time_elapsed >= self.strat_wait_time:
                    # this strategy has had long enough: update records for this strategy, then swap to another strategy
                    self.strats[s]['active'] = False
                    self.strats[s]['profit'] = self.balance
                    self.strats[s]['pps'] = self.profitpertime
                    self.strats[s]['evaluated'] = True

                    new_strat = s + 1
                    if new_strat > self.k - 1:
                        # we've just evaluated the last of this trader's set of strategies
                        # sort the strategies into order of descending profitability
                        strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)

                        # use this as a control: unsorts the strats, gives pure random walk.
                        # strats_sorted = self.strats

                        # the sorted list of strats replaces the existing list
                        self.strats = strats_sorted

                        # signal that we want to record a system snapshot because this trader's eval loop finished
                        snapshot = True

                        # NB not updating self.active_strat here because next call to respond() generates new popln

                    else:
                        # copy the new strategy vector into the trader's params
                        load_strat(self.strats[new_strat]['stratvec'], time)
                        self.strats[new_strat]['start_t'] = time
                        self.active_strat = new_strat
                        self.strats[new_strat]['active'] = True
                        self.last_strat_change_time = time

                    if vrbs and self.tid == 'S00':
                        vstr = 't=%.3f (%.2fdays) %s ZIPSH respond:' % (time, time/86400, self.tid)
                        vstr += ' strat[%d] elapsed=%.3f; wait_t=%.3f, pps=%f' % \
                                (s, time_elapsed, self.strat_wait_time, self.strats[s]['pps'])
                        if new_strat > self.k - 1:
                            print(vstr)
                        else:
                            vstr += ' switching to strat[%d]: %s' %\
                                    (new_strat, self.strat_csv_str(self.strats[new_strat]['stratvec']))

        elif self.optmzr is None:
            # this is vanilla ZIP -- nonadaptive, no optimizer, nothing to change here.
            pass

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if vrbs and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('ZIP respond: B_improved', bid_improved, 'B_hit', bid_hit,
                  'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if self.job == 'Ask':
            # seller
            if deal:
                tradeprice = trade['price']
                if self.price <= tradeprice:
                    # could sell for more? raise margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                tradeprice = trade['price']
                if self.price >= tradeprice:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        # return value of respond() tells caller whether to print a new frame of system-snapshot data
        return snapshot


class TraderPT1(Trader):
    """
    A minimally simple propreitary trader that buys & sells to make profit

    PT1 long-only buy-and-hold strategy in pseudocode:

    1 wait until the market has been open for 5 minutes (to give prices a chance to settle)
    2 then repeat forever:
    2.1 if (I am not holding a unit)
    2.1.1  and (best ask price is "cheap" -- i.e., less than average of recent transaction prices)
    2.1.2  and (I have enough money in my bank to pay the asking price)
    2.2 then
    2.2.1   (buy the unit -- lift the ask)
    2.2.2   (remember the purchase-price I paid for it)
    2.3 else if (I am holding a unit)
    2.4 then
    2.4.1   (my asking-price is that unit's purchase-price plus my profit margin)
    2.4.1   if (best bid price is more than my asking price)
    2.4.1   then
    2.4.1.1    (sell my unit -- hit the bid)
    2.4.1.2    (put the money in my bank)
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Construct a PT1 trader
        :param ttype: the ticker-symbol for the type of trader (its strategy)
        :param tid: the trader id
        :param balance: the trader's bank balance
        :param params: a dictionary of optional parameter-values to override the defaults
        :param time: the current time.
        """
        
        init_verbose = True
        
        Trader.__init__(self, ttype, tid, balance, params, time)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'; shows what PT1 is currently trying to do
        self.last_purchase_price = None

        # Default parameter-values
        self.n_past_trades = 5      # how many recent trades used to compute average price (avg_p)?
        self.bid_percent = 0.9999   # what percentage of avg_p should best_ask be for this trader to bid
        self.ask_delta = 5          # how much (absolute value) to improve on purchase price

        # Did the caller provide different params?
        if type(params) is dict:
            if 'bid_percent' in params:
                self.bid_percent = params['bid_percent']
                if self.bid_percent > 1.0 or self.bid_percent < 0.01:
                    sys.exit('FAIL: self.bid_percent=%f not in range [0.01,1.0])' % self.bid_percent)
            if 'ask_delta' in params:
                self.ask_delta = params['ask_delta']
                if self.ask_delta < 0:
                    sys.exit('Fail: PT1 ask_delta can\'t be negative (it\'s an absolute value)')
            if 'n_past_trades' in params:
                self.n_past_trades = int(round(params['n_past_trades']))
                if self.n_past_trades < 1:
                    sys.exit('Fail: PT1 n_past trades must be 1 or more')
                    
        if init_verbose:
            print('PT1 init: n_past_trades=%d, bid_percent=%6.5f, ask_delta=%d\n'
                  % (self.n_past_trades, self.bid_percent, self.ask_delta))
            
    def getorder(self, time, countdown, lob):
        """
        return this trader's order when it is polled in the main market_session loop.
        :param time: the current time.
        :param countdown: the time remaining until market closes (not currently used).
        :param lob: the public lob.
        :return: trader's new order, or None.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to the current state of the public lob.
        Buys if best bid is less than simple moving average of recent transcaction prices.
        Sells as soon as it can make an acceptable profit.
        :param time: the current time
        :param lob: the current public lob
        :param trade:
        :param vrbs: verbosity -- if True then print running commentary, else stay silent
        :return: <nothing>
        """

        vstr = 't=%f PT1 respond: ' % time

        # what is average price of most recent n trades?
        # work backwards from end of tape (most recent trade)
        tape_position = -1
        n_prices = 0
        sum_prices = 0
        avg_price_ok = False
        avg_price = -1
        while n_prices < self.n_past_trades and abs(tape_position) < len(lob['tape']):
            if lob['tape'][tape_position]['type'] == 'Trade':
                price = lob['tape'][tape_position]['price']
                n_prices += 1
                sum_prices += price
            tape_position -= 1
        if n_prices == self.n_past_trades:
            # there's been enough trades to form an acceptable average
            avg_price = int(round(sum_prices / n_prices))
            avg_price_ok = True
        vstr += "avg_price_ok=%s, avg_price=%d " % (avg_price_ok, avg_price)

        # buying?
        if self.job == 'Buy' and avg_price_ok:
            vstr += 'Buying - '
            # see what's on the LOB
            if lob['asks']['n'] > 0:
                # there is at least one ask on the LOB
                best_ask = lob['asks']['best']
                if best_ask / avg_price < self.bid_percent:
                    # bestask is good value: send a spread-crossing bid to lift the ask
                    bidprice = best_ask + 1
                    if bidprice < self.balance:
                        # can afford to buy
                        # create the bid by issuing order to self, which will be processed in getorder()
                        order = Order(self.tid, 'Bid', bidprice, 1, time, lob['QID'])
                        self.orders = [order]
                        vstr += 'Best ask=%d, bidprice=%d, order=%s ' % (best_ask, bidprice, order)
                else:
                    vstr += 'bestask=%d >= avg_price=%d' % (best_ask, avg_price)
            else:
                vstr += 'No asks on LOB'
        # selling?
        elif self.job == 'Sell':
            vstr += 'Selling - '
            # see what's on the LOB
            if lob['bids']['n'] > 0:
                # there is at least one bid on the LOB
                best_bid = lob['bids']['best']
                # sell single unit at price of purchaseprice+askdelta
                askprice = self.last_purchase_price + self.ask_delta
                if askprice < best_bid:
                    # seems we have a buyer
                    # lift the ask by issuing order to self, which will processed in getorder()
                    order = Order(self.tid, 'Ask', askprice, 1, time, lob['QID'])
                    self.orders = [order]
                    vstr += 'Best bid=%d greater than askprice=%d order=%s ' % (best_bid, askprice, order)
                else:
                    vstr += 'Best bid=%d too low for askprice=%d ' % (best_bid, askprice)
            else:
                vstr += 'No bids on LOB'

        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)

        if vrbs:
            print(vstr)

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records of its bank balance, current orders, and current job
        :param trade: the current time
        :param order: this trader's successful order
        :param vrbs: verbosity -- if True then print running commentary, else stay silent.
        :param time: the current time.
        :return: <nothing>
        """

        # output string outstr is printed if vrbs==True
        mins = int(time//60)
        secs = time - 60 * mins
        hrs = int(mins//60)
        mins = mins - 60 * hrs
        outstr = 't=%f (%dh%02dm%02ds) %s (%s) bookkeep: orders=' % (time, hrs, mins, secs, self.tid, self.ttype)
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            # Bid order succeeded, remember the price and adjust the balance
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.job = 'Sell'  # now try to sell it for a profit
        elif self.orders[0].otype == 'Ask':
            # Sold! put the money in the bank
            self.balance += transactionprice
            self.last_purchase_price = 0
            self.job = 'Buy'  # now go back and buy another one
        else:
            sys.exit('FATAL: PT1 doesn\'t know .otype %s\n' % self.orders[0].otype)

        if vrbs:
            net_worth = self.balance + self.last_purchase_price
            print('%s Balance=%d NetWorth=%d' % (outstr, self.balance, net_worth))

        self.del_order(order)  # delete the order

    # end of PT1 definition


class TraderPT2(Trader):
    """
    A A minimally simple propreitary trader that buys & sells to make profit

    PT2 long-only buy-and-hold strategy in pseudocode:

    1 wait until the market has been open for 5 minutes (to give prices a chance to settle)
    2 then repeat forever:
    2.1 if (I am not holding a unit)
    2.1.1  and (best ask price is "cheap" -- i.e., less than average of recent transaction prices)
    2.1.2  and (I have enough money in my bank to pay the asking price)
    2.2 then
    2.2.1   (buy the unit -- lift the ask)
    2.2.2   (remember the purchase-price I paid for it)
    2.3 else if (I am holding a unit)
    2.4 then
    2.4.1   (my asking-price is that unit's purchase-price plus my profit margin)
    2.4.1   if (best bid price is more than my asking price)
    2.4.1   then
    2.4.1.1    (sell my unit -- hit the bid)
    2.4.1.2    (put the money in my bank)
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Construct a PT2 trader
        :param ttype: the ticker-symbol for the type of trader (its strategy)
        :param tid: the trader id
        :param balance: the trader's bank balance
        :param params: a dictionary of optional parameter-values to override the defaults
        :param time: the current time.
        """

        Trader.__init__(self, ttype, tid, balance, params, time)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'; shows what PT2 is currently trying to do
        self.last_purchase_price = None
        
        init_verbose = True

        # Default parameter-values
        self.n_past_trades = 5      # how many recent trades used to compute average price (avg_p)?
        self.bid_percent = 0.9999   # what percentage of avg_p should best_ask be for this trader to bid
        self.ask_delta = 5          # how much (absolute value) to improve on purchase price

        # Did the caller provide different params?
        if type(params) is dict:
            if 'bid_percent' in params:
                self.bid_percent = params['bid_percent']
                if self.bid_percent > 1.0 or self.bid_percent < 0.01:
                    sys.exit('FAIL: PT2 self.bid_percent=%f not in range [0.01,1.0])' % self.bid_percent)
            if 'ask_delta' in params:
                self.ask_delta = params['ask_delta']
                if self.ask_delta < 0:
                    sys.exit('Fail: PT2 ask_delta can\'t be negative (it\'s an absolute value)')
            if 'n_past_trades' in params:
                self.n_past_trades = int(round(params['n_past_trades']))
                if self.n_past_trades < 1:
                    sys.exit('Fail: PT2 n_past trades must be 1 or more')
                    
        if init_verbose:
            print('PT2 init: n_past_trades=%d, bid_percent=%6.5f, ask_delta=%d\n'
                  % (self.n_past_trades, self.bid_percent, self.ask_delta))

    def getorder(self, time, countdown, lob):
        """
        return this trader's order when it is polled in the main market_session loop.
        :param time: the current time.
        :param countdown: the time remaining until market closes (not currently used).
        :param lob: the public lob.
        :return: trader's new order, or None.
        """
        # this test for negative countdown is purely to stop PyCharm warning about unused parameter value
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to the current state of the public lob.
        Buys if best bid is less than simple moving average of recent transcaction prices.
        Sells as soon as it can make an acceptable profit.
        :param time: the current time
        :param lob: the current public lob
        :param trade:
        :param vrbs: if True then print running commentary, else stay silent
        :return: <nothing>
        """

        vstr = 't=%f PT2 respond: ' % time

        # what is average price of most recent n trades?
        # work backwards from end of tape (most recent trade)
        tape_position = -1
        n_prices = 0
        sum_prices = 0
        avg_price_ok = False
        avg_price = -1
        while n_prices < self.n_past_trades and abs(tape_position) < len(lob['tape']):
            if lob['tape'][tape_position]['type'] == 'Trade':
                price = lob['tape'][tape_position]['price']
                n_prices += 1
                sum_prices += price
            tape_position -= 1
        if n_prices == self.n_past_trades:
            # there's been enough trades to form an acceptable average
            avg_price = int(round(sum_prices / n_prices))
            avg_price_ok = True
        vstr += "avg_price_ok=%s, avg_price=%d " % (avg_price_ok, avg_price)

        # buying?
        if self.job == 'Buy' and avg_price_ok:
            vstr += 'Buying - '
            # see what's on the LOB
            if lob['asks']['n'] > 0:
                # there is at least one ask on the LOB
                best_ask = lob['asks']['best']
                if best_ask / avg_price < self.bid_percent:
                    # bestask is good value: send a spread-crossing bid to lift the ask
                    bidprice = best_ask + 1
                    if bidprice < self.balance:
                        # can afford to buy
                        # create the bid by issuing order to self, which will be processed in getorder()
                        order = Order(self.tid, 'Bid', bidprice, 1, time, lob['QID'])
                        self.orders = [order]
                        vstr += 'Best ask=%d, bidprice=%d, order=%s ' % (best_ask, bidprice, order)
                else:
                    vstr += 'bestask=%d >= avg_price=%d' % (best_ask, avg_price)
            else:
                vstr += 'No asks on LOB'
        # selling?
        elif self.job == 'Sell':
            vstr += 'Selling - '
            # see what's on the LOB
            if lob['bids']['n'] > 0:
                # there is at least one bid on the LOB
                best_bid = lob['bids']['best']
                # sell single unit at price of purchaseprice+askdelta
                askprice = self.last_purchase_price + self.ask_delta
                if askprice < best_bid:
                    # seems we have a buyer
                    # lift the ask by issuing order to self, which will processed in getorder()
                    order = Order(self.tid, 'Ask', askprice, 1, time, lob['QID'])
                    self.orders = [order]
                    vstr += 'Best bid=%d greater than askprice=%d order=%s ' % (best_bid, askprice, order)
                else:
                    vstr += 'Best bid=%d too low for askprice=%d ' % (best_bid, askprice)
            else:
                vstr += 'No bids on LOB'

        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)

        if vrbs:
            print(vstr)

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records of its bank balance, current orders, and current job
        :param trade: the current time
        :param order: this trader's successful order
        :param vrbs: if True then print a running commentary, otherwise stay silent.
        :param time: the current time.
        :return: <nothing>
        """

        # output string outstr is printed if vrbs==True
        mins = int(time//60)
        secs = time - 60 * mins
        hrs = int(mins//60)
        mins = mins - 60 * hrs
        outstr = 't=%f (%dh%02dm%02ds) %s (%s) bookkeep: orders=' % (time, hrs, mins, secs, self.tid, self.ttype)
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            # Bid order succeeded, remember the price and adjust the balance
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.job = 'Sell'  # now try to sell it for a profit
        elif self.orders[0].otype == 'Ask':
            # Sold! put the money in the bank
            self.balance += transactionprice
            self.last_purchase_price = 0
            self.job = 'Buy'  # now go back and buy another one
        else:
            sys.exit('FATAL: PT2 doesn\'t know .otype %s\n' % self.orders[0].otype)

        if vrbs:
            net_worth = self.balance + self.last_purchase_price
            print('%s Balance=%d NetWorth=%d' % (outstr, self.balance, net_worth))

        self.del_order(order)  # delete the order

    # end of PT2 definition

# ########################---trader-types have all been defined now--################

# LLM Trader Classes
class TraderLLMProp(Trader):
    """
    LLM-based proprietary trader that follows the same buy-and-hold strategy pattern as PT1/PT2
    but uses LLM decision making for determining when to buy and sell.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the LLM proprietary trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance (should be same as PT1/PT2)
        :param params: parameters including API key and trading strategy params
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # LLM configuration
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Lower temperature for more consistent trading decisions
        self.max_tokens = 4000
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized LLM prop trader {tid} with model {self.model_name}")
        else:
            print(f"Warning: No API key provided for LLM prop trader {tid}")
            self.model = None
        
        # Proprietary trading state (similar to PT1/PT2)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0  # how many units we currently hold
        
        # Trading history for LLM context
        self.trading_history = []
        self.max_history = 20
        
        # Default trading parameters (can be overridden in params)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def _format_market_context(self, lob, time):
        """
        Format market data for the LLM in a structured way
        """
        # Recent transaction prices for context
        recent_prices = []
        tape_position = -1
        n_prices = 0
        while n_prices < self.n_past_trades and abs(tape_position) < len(lob['tape']):
            if lob['tape'][tape_position]['type'] == 'Trade':
                recent_prices.append(lob['tape'][tape_position]['price'])
                n_prices += 1
            tape_position -= 1
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price else "N/A"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        bid_ask_spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Analyze trader activity
        trade_analysis = self.analyze_recent_trades(lob, n_trades=10)
        trader_activity = trade_analysis['trader_activity']
        
        # Format trader activity information
        trader_info = ""
        if trader_activity:
            trader_info = "RECENT TRADER ACTIVITY:\n"
            for trader_id, activity in trader_activity.items():
                if trader_id != self.tid:  # Don't show our own activity
                    avg_price_str = f"${activity['avg_price']:.1f}" if activity['avg_price'] is not None else "N/A"
                trader_info += f"Trader {trader_id}: {activity['trades']} trades, avg price {avg_price_str}\n"
        
        # Clear state reporting
        context = f"""MARKET DATA:
Time: {time:.1f}
Best Bid: {best_bid}
Best Ask: {best_ask}
Spread: {bid_ask_spread}
Recent prices: {recent_prices}
Average recent price: {avg_price_str}

{trader_info}
MY CURRENT STATE:
Balance: ${self.balance}
Current job: {self.job}
Inventory: {self.inventory} units
Last purchase price: ${self.last_purchase_price if self.last_purchase_price else 'None'}
Number of completed trades: {self.n_trades}

RECENT DECISIONS:
{self._format_recent_history()}
"""
        return context

    def _format_recent_history(self):
        """Format recent trading decisions and state changes for context"""
        if not self.trading_history:
            return "No recent trading history"
        
        history_str = ""
        for entry in self.trading_history[-5:]:  # Last 5 events
            if 'event' in entry and entry['event'] in ['BOUGHT', 'SOLD']:
                # This is a trade execution
                if entry['event'] == 'BOUGHT':
                    history_str += f"Time {entry['time']:.1f}: BOUGHT at ${entry['price']}, switched to job={entry['new_job']}\n"
                else:  # SOLD
                    history_str += f"Time {entry['time']:.1f}: SOLD at ${entry['price']}, profit=${entry['profit']}, switched to job={entry['new_job']}\n"
            else:
                # This is a decision
                history_str += f"Time {entry['time']:.1f}: DECISION={entry['decision']} - {entry['reasoning']}\n"
        return history_str

    def _get_llm_trading_decision(self, market_context):
        """
        Get trading decision from LLM
        """
        if not self.model:
            return self._fallback_decision()
        
    def _get_llm_trading_decision(self, market_context):
        """
        Get trading decision from LLM
        """
        if not self.model:
            return self._fallback_decision()
        
        # Extract market data from context for use in prompts
        avg_price_line = [line for line in market_context.split('\n') if 'Average recent price:' in line]
        avg_price_str = avg_price_line[0].split(': ')[1] if avg_price_line else "N/A"
        
        best_bid_line = [line for line in market_context.split('\n') if 'Best Bid:' in line]
        best_bid_str = best_bid_line[0].split(': ')[1] if best_bid_line else "None"
        
        if self.job == 'Buy':
            prompt = f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

{market_context}

CURRENT SITUATION: You currently have NO INVENTORY and are looking to BUY a unit.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with.

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""

        elif self.job == 'Sell':
            prompt = f"""You are a proprietary trader trying to make profit by buying low and selling high.

{market_context}

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with ($500).

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid_str} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price  
"WAIT" - to wait for better conditions

No explanation needed."""

        else:
            # This shouldn't happen but handle gracefully
            return self._fallback_decision()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            print(f"LLM API error for trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response into actionable decision
        """
        response_upper = response_text.upper()
        
        # Look for explicit decision patterns first (more specific)
        import re
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            # Ensure price is within valid bounds
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            # Ensure price is within valid bounds
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text
            }
        
        # If no explicit price commands found, default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        # Let LLM decide when to start trading - no forced delays
        
        # Get market context and LLM decision
        market_context = self._format_market_context(lob, time)
        decision = self._get_llm_trading_decision(market_context)
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision - let LLM have full control
        if decision['action'] == 'BUY' and self.job == 'Buy':
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            self._execute_sell_decision(decision, lob, time)

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        # Ensure we have a valid price from LLM
        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        # Ensure we have a valid price from LLM
        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade - CRITICAL for state management
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # CRITICAL: Switch to selling mode
            
            print(f"📦 LLM Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                emoji = "🟢" if profit >= 0 else "🔴"
                print(f"{emoji} LLM Trader SOLD at ${transactionprice} | Profit: ${profit} | Balance: ${self.balance}")
            else:
                profit = 0  # Fallback if purchase price is missing
                print(f"🔴 LLM Trader SOLD at ${transactionprice} | No purchase price recorded | Balance: ${self.balance}")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # CRITICAL: Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)

    def analyze_recent_trades(self, lob, n_trades=10):
        """
        Analyze recent trades to understand market activity and trader behavior
        :param lob: the current limit order book
        :param n_trades: number of recent trades to analyze
        :return: dictionary with analysis results
        """
        if not lob['tape']:
            return {'trader_activity': {}, 'trade_patterns': {}}
        
        # Get recent trades
        recent_trades = []
        tape_position = -1
        n_analyzed = 0
        
        while n_analyzed < n_trades and abs(tape_position) < len(lob['tape']):
            if lob['tape'][tape_position]['type'] == 'Trade':
                trade = lob['tape'][tape_position]
                recent_trades.append(trade)
                n_analyzed += 1
            tape_position -= 1
        
        if not recent_trades:
            return {'trader_activity': {}, 'trade_patterns': {}}
        
        # Analyze trader activity
        trader_activity = {}
        trader_prices = {}
        
        for trade in recent_trades:
            # Count trades by trader
            for party in ['party1', 'party2']:
                trader_id = trade[party]
                if trader_id not in trader_activity:
                    trader_activity[trader_id] = {'trades': 0, 'total_volume': 0}
                trader_activity[trader_id]['trades'] += 1
                trader_activity[trader_id]['total_volume'] += trade['qty']
                
                # Track prices by trader
                if trader_id not in trader_prices:
                    trader_prices[trader_id] = []
                trader_prices[trader_id].append(trade['price'])
        
        # Calculate average prices for each trader
        for trader_id in trader_prices:
            avg_price = sum(trader_prices[trader_id]) / len(trader_prices[trader_id])
            trader_activity[trader_id]['avg_price'] = avg_price
            trader_activity[trader_id]['price_range'] = {
                'min': min(trader_prices[trader_id]),
                'max': max(trader_prices[trader_id])
            }
        
        # Analyze trade patterns
        trade_patterns = {
            'total_trades': len(recent_trades),
            'price_trend': 'stable',
            'volume_trend': 'stable'
        }
        
        if len(recent_trades) >= 2:
            # Simple trend analysis
            first_half = recent_trades[:len(recent_trades)//2]
            second_half = recent_trades[len(recent_trades)//2:]
            
            first_avg = sum(t['price'] for t in first_half) / len(first_half)
            second_avg = sum(t['price'] for t in second_half) / len(second_half)
            
            if second_avg > first_avg * 1.02:  # 2% increase
                trade_patterns['price_trend'] = 'increasing'
            elif second_avg < first_avg * 0.98:  # 2% decrease
                trade_patterns['price_trend'] = 'decreasing'
        
        return {
            'trader_activity': trader_activity,
            'trade_patterns': trade_patterns,
            'recent_trades': recent_trades
        }

# Belief Graph Trader Class
class TraderBeliefGraph(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import BeliefGraph, MarketEvent, EventType
            self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 8000  # Maximum for complete reasoning
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            bg_logger.info(f"Initialized BG (Belief Graph + CoT) trader {tid} with model {self.model_name}")
        else:
            bg_logger.warning(f"No API key provided for BG (Belief Graph + CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # DEBUG: Log raw event data from BSE
        bg_logger.debug(f"[BG-RAW-EVENT] {self.tid}: Processing raw event: {event}")
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
            bg_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected TRADE event")
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
            bg_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected BID event")
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
            bg_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected ASK event")
        else:
            bg_logger.debug(f"[BG-EVENT-SKIP] {self.tid}: Skipping unknown event type: {event['type']}")
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # DEBUG: Log parsed market event
        bg_logger.debug(f"[BG-PARSED-EVENT] {self.tid}: Created MarketEvent - Type: {event_type.value}, Agent: {market_event.agent_id}, Price: {market_event.price}")
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)
        bg_logger.debug(f"[BG-UPDATED] {self.tid}: Belief graph updated with {event_type.value} event")

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _generate_natural_language_insights(self):
        """
        Convert the belief graph JSON into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        # Analyze trading patterns
        insights.append("=== MARKET BEHAVIOR ANALYSIS ===")
        
        # Group agents by strategy
        aggressive_agents = []
        passive_agents = []
        neutral_agents = []
        
        for agent_id, node in agents.items():
            if node.strategy_type == "aggressive":
                aggressive_agents.append((agent_id, node))
            elif node.strategy_type == "passive":
                passive_agents.append((agent_id, node))
            else:
                neutral_agents.append((agent_id, node))
        
        # Strategy insights
        if aggressive_agents:
            insights.append(f"AGGRESSIVE TRADERS ({len(aggressive_agents)}): These agents tend to pay premium prices or accept lower selling prices to execute trades quickly.")
            for agent_id, node in aggressive_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        if passive_agents:
            insights.append(f"PASSIVE TRADERS ({len(passive_agents)}): These agents wait for better prices and are more patient.")
            for agent_id, node in passive_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        if neutral_agents:
            insights.append(f"NEUTRAL TRADERS ({len(neutral_agents)}): These agents trade at market prices without strong urgency.")
            for agent_id, node in neutral_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        # Price analysis
        insights.append("\n=== VALUATION PATTERNS ===")
        valuations = [(node.inferred_valuation, agent_id, node.strategy_type) 
                     for agent_id, node in agents.items() 
                     if node.inferred_valuation is not None]
        
        if valuations:
            valuations.sort(reverse=True)  # Highest to lowest
            highest_val = valuations[0]
            lowest_val = valuations[-1]
            
            insights.append(f"Highest valuation: {highest_val[1]} values at ${highest_val[0]} (strategy: {highest_val[2]})")
            insights.append(f"Lowest valuation: {lowest_val[1]} values at ${lowest_val[0]} (strategy: {lowest_val[2]})")
            
            avg_val = sum(v[0] for v in valuations) / len(valuations)
            insights.append(f"Average market valuation: ${avg_val:.1f}")
        
        # Temporal insights from event history
        if hasattr(self.belief_graph, 'event_history') and self.belief_graph.event_history:
            insights.append("\n=== RECENT TRADING SEQUENCE ===")
            recent_events = self.belief_graph.event_history[-5:]  # Last 5 events
            
            for i, event in enumerate(recent_events):
                if event.agent_id in agents:
                    agent_node = agents[event.agent_id]
                    insights.append(f"{i+1}. {event.agent_id} ({agent_node.strategy_type}) traded at ${event.price}")
        
        # Strategic recommendations
        insights.append("\n=== STRATEGIC INSIGHTS ===")
        
        if aggressive_agents and passive_agents:
            avg_aggressive_price = sum(node.last_trade_price for _, node in aggressive_agents) / len(aggressive_agents)
            avg_passive_price = sum(node.last_trade_price for _, node in passive_agents) / len(passive_agents)
            
            if avg_aggressive_price > avg_passive_price:
                insights.append(f"Aggressive traders are paying ${avg_aggressive_price - avg_passive_price:.1f} more on average than passive traders.")
                insights.append("This suggests there may be opportunities to be more patient and get better prices.")
            else:
                insights.append("Aggressive and passive traders are getting similar prices, suggesting a balanced market.")
        
        return "\n".join(insights)

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Think step by step about this trading decision. Use your belief graph insights, market analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- BUY [specific_price] (e.g., BUY 95)  
- WAIT

Your response must end with either "BUY [price]" or "WAIT"."""
        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Think step by step about this selling decision. Use your belief graph insights, profit analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- SELL [specific_price] (e.g., SELL 105)
- WAIT

Your response must end with either "SELL [price]" or "WAIT"."""
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response with CoT into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Extract natural Chain of Thought reasoning
        # Look for decision line and capture everything before it as reasoning
        decision_pattern = r'(.*?)(?:Final decision:|DECISION:|BUY|SELL|WAIT)'
        reasoning_match = re.search(decision_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        natural_reasoning = ""
        if reasoning_match:
            natural_reasoning = reasoning_match.group(1).strip()
        else:
            # If no clear decision marker, use the whole response as reasoning
            natural_reasoning = response_text
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text,
            'natural_cot': natural_reasoning
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        bg_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            bg_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                bg_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                bg_logger.info("="*80)
                bg_logger.info(graph_json)
                bg_logger.info("="*80)
                
            except Exception as e:
                bg_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            bg_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        bg_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        bg_logger.info("="*80)
        bg_logger.info(prompt)
        bg_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        bg_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        bg_logger.info(f"  Action: {decision['action']}")
        bg_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        bg_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log Chain of Thought analysis if available
        if 'natural_cot' in decision and decision['natural_cot']:
            bg_logger.info(f"[BG-COT] {self.tid}: === CHAIN OF THOUGHT REASONING ===")
            bg_logger.info(f"[BG-COT-REASONING] {self.tid}: {decision['natural_cot']}")
        
        # Log market context for analysis
        bg_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        bg_logger.info(f"  Current Job: {self.job}")
        bg_logger.info(f"  Balance: ${self.balance}")
        bg_logger.info(f"  Inventory: {self.inventory}")
        bg_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        bg_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        bg_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        bg_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            bg_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            bg_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            bg_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            bg_logger.info(f"📦 BG Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                bg_logger.info(f"{emoji} BG Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                print(f"{emoji} BG Trader {self.tid} SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
            else:
                profit = 0
                bg_logger.info(f"🔴 BG Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 BG Trader {self.tid} SOLD at ${transactionprice} | No purchase price recorded")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)

class GraphVar1(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    
    GraphVar1: Uses DISCRETE belief sets with set elimination logic
    - Maintains possible values for each belief (e.g., possible_valuations: [85, 90, 95])
    - Updates via Hanabi-style elimination based on observations
    - Reduces possibility space as more evidence accumulates
    
    NOTE: Fixed issue with belief graph not updating due to timestamp logic bug.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import GraphVar1, MarketEvent, EventType
            self.belief_graph = GraphVar1(asset_id="BSE_ASSET")
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 8000  # Maximum for complete reasoning
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            gv1_logger.info(f"Initialized BG (Belief Graph + CoT) trader {tid} with model {self.model_name}")
        else:
            gv1_logger.warning(f"No API key provided for BG (Belief Graph + CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # DEBUG: Log raw event data from BSE
        gv1_logger.debug(f"[BG-RAW-EVENT] {self.tid}: Processing raw event: {event}")
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
            gv1_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected TRADE event")
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
            gv1_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected BID event")
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
            gv1_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected ASK event")
        else:
            gv1_logger.debug(f"[BG-EVENT-SKIP] {self.tid}: Skipping unknown event type: {event['type']}")
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # DEBUG: Log parsed market event
        gv1_logger.debug(f"[BG-PARSED-EVENT] {self.tid}: Created MarketEvent - Type: {event_type.value}, Agent: {market_event.agent_id}, Price: {market_event.price}")
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)
        gv1_logger.info(f"[BG-UPDATED] {self.tid}: Belief graph updated with {event_type.value} event from {market_event.agent_id} at price {market_event.price}")
        
        # Log the current belief state for this agent (if exists)
        if market_event.agent_id and market_event.agent_id in self.belief_graph.nodes:
            agent_beliefs = self.belief_graph.get_agent_beliefs(market_event.agent_id)
            if agent_beliefs:
                val_est = agent_beliefs.get('valuation_estimate', 'Unknown')
                strat = agent_beliefs.get('strategy_type', 'Unknown')
                gv1_logger.debug(f"[BG-BELIEFS] {self.tid}: Agent {market_event.agent_id} beliefs - valuation={val_est}, strategy={strat}")

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _generate_natural_language_insights(self):
        """
        Convert the discrete belief graph into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        insights.append("=== DISCRETE BELIEF ANALYSIS ===")
        
        # Analyze each agent's narrowed belief sets
        for agent_id in agents.keys():
            agent_insights = []
            agent_insights.append(f"\n--- Agent {agent_id} Beliefs ---")
            
            # Get valuation beliefs
            valuation_edge = self._get_belief_edge(agent_id, "valuation")
            if valuation_edge and valuation_edge.value and "possible_valuations" in valuation_edge.value:
                possible_vals = valuation_edge.value["possible_valuations"]
                if len(possible_vals) < 9:  # Narrowed from initial 9 values
                    val_range = f"${min(possible_vals)}-${max(possible_vals)}"
                    agent_insights.append(f"  • Valuation NARROWED to {len(possible_vals)} possibilities: {val_range}")
                else:
                    agent_insights.append(f"  • Valuation: Full range ${min(possible_vals)}-${max(possible_vals)} (no narrowing yet)")
            
            # Get market direction beliefs
            direction_edge = self._get_belief_edge(agent_id, "market_direction")
            if direction_edge and "direction_distribution" in direction_edge.value:
                possible_dirs = direction_edge.value["direction_distribution"]
                if len(possible_dirs) < 3:  # Narrowed from initial 3 values
                    agent_insights.append(f"  • Market Direction NARROWED to: {', '.join(possible_dirs)}")
                else:
                    agent_insights.append(f"  • Market Direction: All possibilities (up, down, sideways)")
            
            # Get desperation level beliefs
            desperation_edge = self._get_belief_edge(agent_id, "desperation_level")
            if desperation_edge and "desperation_distribution" in desperation_edge.value:
                possible_desp = desperation_edge.value["desperation_distribution"]
                if len(possible_desp) < 3:  # Narrowed from initial 3 values
                    agent_insights.append(f"  • Desperation NARROWED to: {', '.join(possible_desp)}")
                else:
                    agent_insights.append(f"  • Desperation: All levels possible (calm, moderate, desperate)")
            
            # Get available cash beliefs
            cash_edge = self._get_belief_edge(agent_id, "available_cash")
            if cash_edge and "cash_distribution" in cash_edge.value:
                cash_distribution = cash_edge.value["cash_distribution"]
                if len(cash_distribution) < 3:  # Narrowed from initial 3 values
                    agent_insights.append(f"  • Available Cash NARROWED to: {', '.join(cash_distribution)}")
                else:
                    agent_insights.append(f"  • Available Cash: All levels possible (low, medium, high)")
            
            # Get exit strategy beliefs
            exit_edge = self._get_belief_edge(agent_id, "exit_strategy")
            if exit_edge and "exit_distribution" in exit_edge.value:
                exit_distribution = exit_edge.value["exit_distribution"]
                if len(exit_distribution) < 3:  # Narrowed from initial 3 values
                    agent_insights.append(f"  • Exit Strategy NARROWED to: {', '.join(exit_distribution)}")
                else:
                    agent_insights.append(f"  • Exit Strategy: All strategies possible")
            
            insights.extend(agent_insights)
        
        # Market-wide patterns from narrowed beliefs
        insights.append("\n=== MARKET PATTERNS FROM BELIEF NARROWING ===")
        
        # Count agents with narrowed valuations
        narrowed_valuations = 0
        high_valuers = 0
        low_valuers = 0
        
        for agent_id in agents.keys():
            val_edge = self._get_belief_edge(agent_id, "valuation")
            if val_edge and val_edge.value and "possible_valuations" in val_edge.value:
                possible_vals = val_edge.value["possible_valuations"]
                if len(possible_vals) < 9:
                    narrowed_valuations += 1
                    avg_val = sum(possible_vals) / len(possible_vals)
                    if avg_val > 100:
                        high_valuers += 1
                    elif avg_val < 100:
                        low_valuers += 1
        
        if narrowed_valuations > 0:
            insights.append(f"• {narrowed_valuations} agents have narrowed valuation beliefs")
            insights.append(f"• {high_valuers} agents likely value asset ABOVE $100")
            insights.append(f"• {low_valuers} agents likely value asset BELOW $100")
        
        # Count agents with narrowed desperation
        desperate_agents = 0
        calm_agents = 0
        
        for agent_id in agents.keys():
            desp_edge = self._get_belief_edge(agent_id, "desperation_level")
            if desp_edge and "desperation_distribution" in desp_edge.value:
                possible_desp = desp_edge.value["desperation_distribution"]
                if len(possible_desp) == 1:
                    if "desperate" in possible_desp:
                        desperate_agents += 1
                    elif "calm" in possible_desp:
                        calm_agents += 1
        
        if desperate_agents > 0 or calm_agents > 0:
            insights.append(f"• {desperate_agents} agents confirmed as DESPERATE (likely to pay premium)")
            insights.append(f"• {calm_agents} agents confirmed as CALM (likely to wait for better prices)")
        
        # Strategic recommendations based on belief narrowing
        insights.append("\n=== STRATEGIC RECOMMENDATIONS ===")
        
        if high_valuers > low_valuers:
            insights.append("• BULLISH SIGNAL: More agents likely value asset above $100")
            insights.append("• Consider pricing orders slightly higher")
        elif low_valuers > high_valuers:
            insights.append("• BEARISH SIGNAL: More agents likely value asset below $100")
            insights.append("• Consider more competitive pricing")
        
        if desperate_agents > calm_agents:
            insights.append("• SELLER OPPORTUNITY: Desperate agents may pay premium prices")
        elif calm_agents > desperate_agents:
            insights.append("• BUYER OPPORTUNITY: Calm agents may wait, creating bid opportunities")
        
        return "\n".join(insights)

    def _get_belief_edge(self, agent_id, belief_type):
        """
        Helper method to get a specific belief edge for an agent
        """
        if not self.belief_graph or not hasattr(self.belief_graph, 'edges'):
            return None
        
        for edge in self.belief_graph.edges.values():
            if (edge.target_node == agent_id and 
                edge.belief_type == belief_type):
                return edge
        
        return None

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Think step by step about this trading decision. Use your belief graph insights, market analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- BUY [specific_price] (e.g., BUY 95)  
- WAIT

Your response must end with either "BUY [price]" or "WAIT"."""
        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Think step by step about this selling decision. Use your belief graph insights, profit analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- SELL [specific_price] (e.g., SELL 105)
- WAIT

Your response must end with either "SELL [price]" or "WAIT"."""
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response with CoT into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Extract natural Chain of Thought reasoning
        # Look for decision line and capture everything before it as reasoning
        decision_pattern = r'(.*?)(?:Final decision:|DECISION:|BUY|SELL|WAIT)'
        reasoning_match = re.search(decision_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        natural_reasoning = ""
        if reasoning_match:
            natural_reasoning = reasoning_match.group(1).strip()
        else:
            # If no clear decision marker, use the whole response as reasoning
            natural_reasoning = response_text
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text,
            'natural_cot': natural_reasoning
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        gv1_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            gv1_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                gv1_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                gv1_logger.info("="*80)
                gv1_logger.info(graph_json)
                gv1_logger.info("="*80)
                
            except Exception as e:
                gv1_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            gv1_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        gv1_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        gv1_logger.info("="*80)
        gv1_logger.info(prompt)
        gv1_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        gv1_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        gv1_logger.info(f"  Action: {decision['action']}")
        gv1_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        gv1_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log Chain of Thought analysis if available
        if 'natural_cot' in decision and decision['natural_cot']:
            gv1_logger.info(f"[BG-COT] {self.tid}: === CHAIN OF THOUGHT REASONING ===")
            gv1_logger.info(f"[BG-COT-REASONING] {self.tid}: {decision['natural_cot']}")
        
        # Log market context for analysis
        gv1_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        gv1_logger.info(f"  Current Job: {self.job}")
        gv1_logger.info(f"  Balance: ${self.balance}")
        gv1_logger.info(f"  Inventory: {self.inventory}")
        gv1_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        gv1_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        gv1_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        gv1_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            gv1_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            gv1_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            gv1_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            gv1_logger.info(f"📦 BG Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            print(f"💰 {self.ttype} {self.tid} BOUGHT at ${transactionprice} | Balance: ${self.balance:.0f}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                gv1_logger.info(f"{emoji} BG Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                if profit > 0:
                    print(f"🟢 {self.ttype} {self.tid} SOLD at ${transactionprice} | Profit: ${profit:.0f} | Balance: ${self.balance:.0f}")
                elif profit == 0:
                    print(f"🟡 {self.ttype} {self.tid} SOLD at ${transactionprice} | Break-even | Balance: ${self.balance:.0f}")
                else:
                    print(f"🔴 {self.ttype} {self.tid} SOLD at ${transactionprice} | Loss: ${abs(profit):.0f} | Balance: ${self.balance:.0f}")
            else:
                profit = 0
                gv1_logger.info(f"🔴 BG Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 {self.ttype} {self.tid} SOLD at ${transactionprice} | No purchase price recorded | Balance: ${self.balance:.0f}")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)

class GraphVar2(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    
    GraphVar2: Uses PROBABILISTIC belief distributions with Bayesian updates
    - Maintains probability distributions for beliefs (e.g., valuation_distribution: {85: 0.4, 90: 0.3, 95: 0.3})
    - Updates via Bayesian inference based on observations
    - Adjusts probabilities as more evidence accumulates
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import GraphVar2, MarketEvent, EventType
            self.belief_graph = GraphVar2(asset_id="BSE_ASSET")
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 8000  # Maximum for complete reasoning
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            gv2_logger.info(f"Initialized BG (Belief Graph + CoT) trader {tid} with model {self.model_name}")
        else:
            gv2_logger.warning(f"No API key provided for BG (Belief Graph + CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # DEBUG: Log raw event data from BSE
        gv2_logger.debug(f"[BG-RAW-EVENT] {self.tid}: Processing raw event: {event}")
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
            gv2_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected TRADE event")
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
            gv2_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected BID event")
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
            gv2_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected ASK event")
        else:
            gv2_logger.debug(f"[BG-EVENT-SKIP] {self.tid}: Skipping unknown event type: {event['type']}")
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # DEBUG: Log parsed market event
        gv2_logger.debug(f"[BG-PARSED-EVENT] {self.tid}: Created MarketEvent - Type: {event_type.value}, Agent: {market_event.agent_id}, Price: {market_event.price}")
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)
        gv2_logger.info(f"[BG-UPDATED] {self.tid}: Belief graph updated with {event_type.value} event from {market_event.agent_id} at price {market_event.price}")
        
        # Log the current belief state for this agent (if exists)
        if market_event.agent_id and market_event.agent_id in self.belief_graph.nodes:
            agent_beliefs = self.belief_graph.get_agent_beliefs(market_event.agent_id)
            if agent_beliefs:
                val_est = agent_beliefs.get('valuation_estimate', 'Unknown')
                strat = agent_beliefs.get('strategy_type', 'Unknown')
                gv2_logger.debug(f"[BG-BELIEFS] {self.tid}: Agent {market_event.agent_id} beliefs - valuation={val_est}, strategy={strat}")

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _generate_natural_language_insights(self):
        """
        Convert the discrete belief graph into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        insights.append("=== PROBABILISTIC BELIEF ANALYSIS ===")
        
        # Analyze each agent's narrowed belief sets
        for agent_id in agents.keys():
            agent_insights = []
            agent_insights.append(f"\n--- Agent {agent_id} Beliefs ---")
            
            # Get valuation beliefs
            valuation_edge = self._get_belief_edge(agent_id, "valuation")
            if valuation_edge and valuation_edge.value and "valuation_distribution" in valuation_edge.value:
                val_dist = valuation_edge.value["valuation_distribution"]
                # Find the highest probability values
                sorted_vals = sorted(val_dist.items(), key=lambda x: x[1], reverse=True)
                top_vals = sorted_vals[:3]  # Top 3 most probable values
                top_probs_str = ", ".join([f"${v}({p:.1%})" for v, p in top_vals])
                agent_insights.append(f"  • Valuation Distribution (top probabilities): {top_probs_str}")            
            # Get market direction beliefs
            direction_edge = self._get_belief_edge(agent_id, "market_direction")
            if direction_edge and "direction_distribution" in direction_edge.value:
                dir_dist = direction_edge.value["direction_distribution"]
                # Show top probability directions
                sorted_dirs = sorted(dir_dist.items(), key=lambda x: x[1], reverse=True)
                top_dir = sorted_dirs[0] if sorted_dirs else None
                if top_dir and top_dir[1] > 0.4:  # Strong preference
                    agent_insights.append(f"  • Market Direction: Strong {top_dir[0]} bias ({top_dir[1]:.1%})")
                else:
                    dir_probs_str = ", ".join([f"{d}({p:.1%})" for d, p in sorted_dirs])
                    agent_insights.append(f"  • Market Direction Distribution: {dir_probs_str}")            
            # Get desperation level beliefs
            desperation_edge = self._get_belief_edge(agent_id, "desperation_level")
            if desperation_edge and "desperation_distribution" in desperation_edge.value:
                desp_dist = desperation_edge.value["desperation_distribution"]
                # Show top probability desperation levels
                sorted_desp = sorted(desp_dist.items(), key=lambda x: x[1], reverse=True)
                top_desp = sorted_desp[0] if sorted_desp else None
                if top_desp and top_desp[1] > 0.5:  # Strong preference
                    agent_insights.append(f"  • Desperation: Mostly {top_desp[0]} ({top_desp[1]:.1%})")
                else:
                    desp_probs_str = ", ".join([f"{d}({p:.1%})" for d, p in sorted_desp])
                    agent_insights.append(f"  • Desperation Distribution: {desp_probs_str}")
            
            # Get available cash beliefs
            cash_edge = self._get_belief_edge(agent_id, "available_cash")
            if cash_edge and "cash_distribution" in cash_edge.value:
                cash_dist = cash_edge.value["cash_distribution"]
                # Show top probability cash levels
                sorted_cash = sorted(cash_dist.items(), key=lambda x: x[1], reverse=True)
                top_cash = sorted_cash[0] if sorted_cash else None
                if top_cash and top_cash[1] > 0.5:  # Strong preference
                    agent_insights.append(f"  • Available Cash: Likely {top_cash[0]} ({top_cash[1]:.1%})")
                else:
                    cash_probs_str = ", ".join([f"{c}({p:.1%})" for c, p in sorted_cash])
                    agent_insights.append(f"  • Cash Distribution: {cash_probs_str}")
            
            # Get exit strategy beliefs
            exit_edge = self._get_belief_edge(agent_id, "exit_strategy")
            if exit_edge and "exit_distribution" in exit_edge.value:
                exit_dist = exit_edge.value["exit_distribution"]
                # Show top probability exit strategies
                sorted_exit = sorted(exit_dist.items(), key=lambda x: x[1], reverse=True)
                top_exit = sorted_exit[0] if sorted_exit else None
                if top_exit and top_exit[1] > 0.5:  # Strong preference
                    agent_insights.append(f"  • Exit Strategy: Prefers {top_exit[0]} ({top_exit[1]:.1%})")
                else:
                    exit_probs_str = ", ".join([f"{e}({p:.1%})" for e, p in sorted_exit])
                    agent_insights.append(f"  • Exit Strategy Distribution: {exit_probs_str}")
            
            insights.extend(agent_insights)
        
        # Market-wide patterns from narrowed beliefs
        insights.append("\n=== MARKET PATTERNS FROM BELIEF NARROWING ===")
        
        # Count agents with narrowed valuations
        narrowed_valuations = 0
        high_valuers = 0
        low_valuers = 0
        
        for agent_id in agents.keys():
            val_edge = self._get_belief_edge(agent_id, "valuation")
            if val_edge and val_edge.value and "possible_valuations" in val_edge.value:
                possible_vals = val_edge.value["possible_valuations"]
            if val_edge and val_edge.value and "valuation_distribution" in val_edge.value:
                val_dist = val_edge.value["valuation_distribution"]
                # Find peak probability value
                sorted_vals = sorted(val_dist.items(), key=lambda x: x[1], reverse=True)
                peak_val = int(sorted_vals[0][0]) if sorted_vals else 100
                if peak_val > 100:
                    high_valuers += 1
                elif peak_val < 100:
                    low_valuers += 1
                narrowed_valuations += 1
        
        insights.append(f"• {narrowed_valuations} agents have updated valuation beliefs")
        if high_valuers > 0:
            insights.append(f"• {high_valuers} agents likely value asset ABOVE $100")
        if low_valuers > 0:
            insights.append(f"• {low_valuers} agents likely value asset BELOW $100")
        
        # Count agents with narrowed desperation
        desperate_agents = 0
        calm_agents = 0
        
        for agent_id in agents.keys():
            desp_edge = self._get_belief_edge(agent_id, "desperation_level")
            if desp_edge and "desperation_distribution" in desp_edge.value:
                possible_desp = desp_edge.value["desperation_distribution"]
                if len(possible_desp) == 1:
                    if "desperate" in possible_desp:
                        desperate_agents += 1
                    elif "calm" in possible_desp:
                        calm_agents += 1
        
        if desperate_agents > 0 or calm_agents > 0:
            insights.append(f"• {desperate_agents} agents confirmed as DESPERATE (likely to pay premium)")
            insights.append(f"• {calm_agents} agents confirmed as CALM (likely to wait for better prices)")
        
        # Strategic recommendations based on belief narrowing
        insights.append("\n=== STRATEGIC RECOMMENDATIONS ===")
        
        if high_valuers > low_valuers:
            insights.append("• BULLISH SIGNAL: More agents likely value asset above $100")
            insights.append("• Consider pricing orders slightly higher")
        elif low_valuers > high_valuers:
            insights.append("• BEARISH SIGNAL: More agents likely value asset below $100")
            insights.append("• Consider more competitive pricing")
        
        if desperate_agents > calm_agents:
            insights.append("• SELLER OPPORTUNITY: Desperate agents may pay premium prices")
        elif calm_agents > desperate_agents:
            insights.append("• BUYER OPPORTUNITY: Calm agents may wait, creating bid opportunities")
        
        return "\n".join(insights)

    def _get_belief_edge(self, agent_id, belief_type):
        """
        Helper method to get a specific belief edge for an agent
        """
        if not self.belief_graph or not hasattr(self.belief_graph, 'edges'):
            return None
        
        for edge in self.belief_graph.edges.values():
            if (edge.target_node == agent_id and 
                edge.belief_type == belief_type):
                return edge
        
        return None

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Think step by step about this trading decision. Use your belief graph insights, market analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- BUY [specific_price] (e.g., BUY 95)  
- WAIT

Your response must end with either "BUY [price]" or "WAIT"."""
        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Think step by step about this selling decision. Use your belief graph insights, profit analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- SELL [specific_price] (e.g., SELL 105)
- WAIT

Your response must end with either "SELL [price]" or "WAIT"."""
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response with CoT into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Extract natural Chain of Thought reasoning
        # Look for decision line and capture everything before it as reasoning
        decision_pattern = r'(.*?)(?:Final decision:|DECISION:|BUY|SELL|WAIT)'
        reasoning_match = re.search(decision_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        natural_reasoning = ""
        if reasoning_match:
            natural_reasoning = reasoning_match.group(1).strip()
        else:
            # If no clear decision marker, use the whole response as reasoning
            natural_reasoning = response_text
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text,
            'natural_cot': natural_reasoning
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        gv2_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            gv2_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                gv2_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                gv2_logger.info("="*80)
                gv2_logger.info(graph_json)
                gv2_logger.info("="*80)
                
            except Exception as e:
                gv2_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            gv2_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        gv2_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        gv2_logger.info("="*80)
        gv2_logger.info(prompt)
        gv2_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        gv2_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        gv2_logger.info(f"  Action: {decision['action']}")
        gv2_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        gv2_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log Chain of Thought analysis if available
        if 'natural_cot' in decision and decision['natural_cot']:
            gv2_logger.info(f"[BG-COT] {self.tid}: === CHAIN OF THOUGHT REASONING ===")
            gv2_logger.info(f"[BG-COT-REASONING] {self.tid}: {decision['natural_cot']}")
        
        # Log market context for analysis
        gv2_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        gv2_logger.info(f"  Current Job: {self.job}")
        gv2_logger.info(f"  Balance: ${self.balance}")
        gv2_logger.info(f"  Inventory: {self.inventory}")
        gv2_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        gv2_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        gv2_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        gv2_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            gv2_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            gv2_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            gv2_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            gv2_logger.info(f"📦 BG Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            print(f"💰 {self.ttype} {self.tid} BOUGHT at ${transactionprice} | Balance: ${self.balance:.0f}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                gv2_logger.info(f"{emoji} BG Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                if profit > 0:
                    print(f"🟢 {self.ttype} {self.tid} SOLD at ${transactionprice} | Profit: ${profit:.0f} | Balance: ${self.balance:.0f}")
                elif profit == 0:
                    print(f"🟡 {self.ttype} {self.tid} SOLD at ${transactionprice} | Break-even | Balance: ${self.balance:.0f}")
                else:
                    print(f"🔴 {self.ttype} {self.tid} SOLD at ${transactionprice} | Loss: ${abs(profit):.0f} | Balance: ${self.balance:.0f}")
            else:
                profit = 0
                gv2_logger.info(f"🔴 BG Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 {self.ttype} {self.tid} SOLD at ${transactionprice} | No purchase price recorded | Balance: ${self.balance:.0f}")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)















class TraderBeliefGraphWithoutCOT(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import BeliefGraph, MarketEvent, EventType
            self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 4000   # Same as LLM trader
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            bgno_logger.info(f"Initialized BGNO (Belief Graph + No CoT) trader {tid} with model {self.model_name}")
        else:
            bgno_logger.warning(f"No API key provided for BGNO (Belief Graph + No CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # DEBUG: Log raw event data from BSE
        bgno_logger.debug(f"[BG-RAW-EVENT] {self.tid}: Processing raw event: {event}")
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
            bgno_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected TRADE event")
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
            bgno_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected BID event")
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
            bgno_logger.debug(f"[BG-EVENT-TYPE] {self.tid}: Detected ASK event")
        else:
            bgno_logger.debug(f"[BG-EVENT-SKIP] {self.tid}: Skipping unknown event type: {event['type']}")
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # DEBUG: Log parsed market event
        bgno_logger.debug(f"[BG-PARSED-EVENT] {self.tid}: Created MarketEvent - Type: {event_type.value}, Agent: {market_event.agent_id}, Price: {market_event.price}")
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)
        bgno_logger.debug(f"[BG-UPDATED] {self.tid}: Belief graph updated with {event_type.value} event")

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _generate_natural_language_insights(self):
        """
        Convert the belief graph JSON into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        # Analyze trading patterns
        insights.append("=== MARKET BEHAVIOR ANALYSIS ===")
        
        # Group agents by strategy
        aggressive_agents = []
        passive_agents = []
        neutral_agents = []
        
        for agent_id, node in agents.items():
            if node.strategy_type == "aggressive":
                aggressive_agents.append((agent_id, node))
            elif node.strategy_type == "passive":
                passive_agents.append((agent_id, node))
            else:
                neutral_agents.append((agent_id, node))
        
        # Strategy insights
        if aggressive_agents:
            insights.append(f"AGGRESSIVE TRADERS ({len(aggressive_agents)}): These agents tend to pay premium prices or accept lower selling prices to execute trades quickly.")
            for agent_id, node in aggressive_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        if passive_agents:
            insights.append(f"PASSIVE TRADERS ({len(passive_agents)}): These agents wait for better prices and are more patient.")
            for agent_id, node in passive_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        if neutral_agents:
            insights.append(f"NEUTRAL TRADERS ({len(neutral_agents)}): These agents trade at market prices without strong urgency.")
            for agent_id, node in neutral_agents:
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around ${node.inferred_valuation} (confidence: {node.valuation_confidence:.0%})")
        
        # Price analysis
        insights.append("\n=== VALUATION PATTERNS ===")
        valuations = [(node.inferred_valuation, agent_id, node.strategy_type) 
                     for agent_id, node in agents.items() 
                     if node.inferred_valuation is not None]
        
        if valuations:
            valuations.sort(reverse=True)  # Highest to lowest
            highest_val = valuations[0]
            lowest_val = valuations[-1]
            
            insights.append(f"Highest valuation: {highest_val[1]} values at ${highest_val[0]} (strategy: {highest_val[2]})")
            insights.append(f"Lowest valuation: {lowest_val[1]} values at ${lowest_val[0]} (strategy: {lowest_val[2]})")
            
            avg_val = sum(v[0] for v in valuations) / len(valuations)
            insights.append(f"Average market valuation: ${avg_val:.1f}")
        
        # Temporal insights from event history
        if hasattr(self.belief_graph, 'event_history') and self.belief_graph.event_history:
            insights.append("\n=== RECENT TRADING SEQUENCE ===")
            recent_events = self.belief_graph.event_history[-5:]  # Last 5 events
            
            for i, event in enumerate(recent_events):
                if event.agent_id in agents:
                    agent_node = agents[event.agent_id]
                    insights.append(f"{i+1}. {event.agent_id} ({agent_node.strategy_type}) traded at ${event.price}")
        
        # Strategic recommendations
        insights.append("\n=== STRATEGIC INSIGHTS ===")
        
        if aggressive_agents and passive_agents:
            avg_aggressive_price = sum(node.last_trade_price for _, node in aggressive_agents) / len(aggressive_agents)
            avg_passive_price = sum(node.last_trade_price for _, node in passive_agents) / len(passive_agents)
            
            if avg_aggressive_price > avg_passive_price:
                insights.append(f"Aggressive traders are paying ${avg_aggressive_price - avg_passive_price:.1f} more on average than passive traders.")
                insights.append("This suggests there may be opportunities to be more patient and get better prices.")
            else:
                insights.append("Aggressive and passive traders are getting similar prices, suggesting a balanced market.")
        
        return "\n".join(insights)

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response with CoT into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Extract natural Chain of Thought reasoning
        # Look for decision line and capture everything before it as reasoning
        decision_pattern = r'(.*?)(?:Final decision:|DECISION:|BUY|SELL|WAIT)'
        reasoning_match = re.search(decision_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        natural_reasoning = ""
        if reasoning_match:
            natural_reasoning = reasoning_match.group(1).strip()
        else:
            # If no clear decision marker, use the whole response as reasoning
            natural_reasoning = response_text
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text,
                'natural_cot': natural_reasoning
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text,
            'natural_cot': natural_reasoning
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        bgno_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            bgno_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                bgno_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                bgno_logger.info("="*80)
                bgno_logger.info(graph_json)
                bgno_logger.info("="*80)
                
            except Exception as e:
                bgno_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            bgno_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        bgno_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        bgno_logger.info("="*80)
        bgno_logger.info(prompt)
        bgno_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        bgno_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        bgno_logger.info(f"  Action: {decision['action']}")
        bgno_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        bgno_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log Chain of Thought analysis if available
        if 'natural_cot' in decision and decision['natural_cot']:
            bgno_logger.info(f"[BG-COT] {self.tid}: === CHAIN OF THOUGHT REASONING ===")
            bgno_logger.info(f"[BG-COT-REASONING] {self.tid}: {decision['natural_cot']}")
        
        # Log market context for analysis
        bgno_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        bgno_logger.info(f"  Current Job: {self.job}")
        bgno_logger.info(f"  Balance: ${self.balance}")
        bgno_logger.info(f"  Inventory: {self.inventory}")
        bgno_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        bgno_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        bgno_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        bgno_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            bgno_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            bgno_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            bgno_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            bgno_logger.info(f"📦 BGNO Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                bgno_logger.info(f"{emoji} BGNO Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                print(f"{emoji} BGNO Trader {self.tid} SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
            else:
                profit = 0
                bgno_logger.info(f"🔴 BGNO Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 BGNO Trader {self.tid} SOLD at ${transactionprice} | No purchase price recorded")
        
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)


class TraderPerfectGraphWithCoT(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import PerfectBeliefGraph, MarketEvent, EventType
            self.belief_graph = PerfectBeliefGraph(asset_id="BSE_ASSET", traders_dict=None)  # Will be set later
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 8000  # Maximum for complete reasoning
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            pgco_logger.info(f"Initialized PGCO (Perfect Graph + CoT) trader {tid} with model {self.model_name}")
        else:
            pgco_logger.warning(f"No API key provided for PGCO (Perfect Graph + CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def set_traders_dict(self, traders_dict):
        """
        Set the traders dictionary for perfect belief graph access.
        This must be called after all traders are created.
        """
        if self.belief_graph is not None:
            self.belief_graph.traders_dict = traders_dict

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
        else:
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        else:
            # Fallback: extract directly from belief data if strategic insights are empty
            competitors_text = "COMPETITOR ANALYSIS:\n"
            competitors_found = False
            if 'beliefs' in belief_context:
                agents_valuations = {}
                for belief in belief_context['beliefs']:
                    if (belief['belief_type'] == 'valuation' and 
                        belief['target_node'] != self.tid and 
                        belief['value'] is not None):
                        agent_id = belief['target_node']
                        agents_valuations[agent_id] = {
                            'valuation': belief['value'],
                            'confidence': belief['confidence']
                        }
                
                for agent_id, data in agents_valuations.items():
                    valuation_str = f"${data['valuation']:.1f}"
                    competitors_text += f"- {agent_id}: Strategy=unknown, "
                    competitors_text += f"Aggressiveness=0.00, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {data['confidence']:.2f})\n"
                    competitors_found = True
            
            if not competitors_found:
                competitors_text += "- No competitor data available\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Think step by step about this trading decision. Use your belief graph insights, market analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- BUY [specific_price] (e.g., BUY 95)  
- WAIT

Your response must end with either "BUY [price]" or "WAIT"."""        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Think step by step about this selling decision. Use your belief graph insights, profit analysis, and trading principles to reason through your choice.

After your analysis, make your final decision. Output ONLY one of:
- SELL [specific_price] (e.g., SELL 105)
- WAIT

Your response must end with either "SELL [price]" or "WAIT"."""
        
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            pgco_logger.warning(f"[BG-NO-MODEL] {self.tid}: No LLM model available, using fallback")
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            
            # Log the raw LLM response
            pgco_logger.info(f"[BG-LLM-RESPONSE] {self.tid}: === RAW LLM RESPONSE ===")
            pgco_logger.info("-"*50)
            pgco_logger.info(response_text)
            pgco_logger.info("-"*50)
            
            parsed_decision = self._parse_llm_response(response_text)
            
            # Log the parsed decision
            pgco_logger.info(f"[BG-PARSED-DECISION] {self.tid}: Parsed decision:")
            pgco_logger.info(f"  Action: {parsed_decision['action']}")
            pgco_logger.info(f"  Price: {parsed_decision.get('price', 'N/A')}")
            pgco_logger.info(f"  Full Reasoning: {parsed_decision.get('reasoning', 'N/A')}")
            
            return parsed_decision
            
        except Exception as e:
            pgco_logger.error(f"[BG-LLM-ERROR] {self.tid}: LLM API error: {e}")
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        pgco_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            pgco_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                pgco_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                pgco_logger.info("="*80)
                pgco_logger.info(graph_json)
                pgco_logger.info("="*80)
                
                # Also log a human-readable summary
                market_summary = self.belief_graph.get_market_summary()
                pgco_logger.info(f"[BG-GRAPH-SUMMARY] {self.tid}: Market Summary:")
                pgco_logger.info(f"  Active Agents: {market_summary['active_agents']}")
                pgco_logger.info(f"  Total Beliefs: {market_summary['total_beliefs']}")
                pgco_logger.info(f"  Recent Events: {market_summary['recent_events']}")
                
                # Log beliefs about each agent
                for node_id, node in self.belief_graph.nodes.items():
                    if hasattr(node, 'agent_id') and node_id != self.tid:  # Skip self
                        agent_beliefs = self.belief_graph.get_agent_beliefs(node_id)
                        pgco_logger.info(f"[BG-AGENT-BELIEFS] {self.tid}: Beliefs about {node_id}:")
                        for belief_type, beliefs in agent_beliefs.items():
                            for belief in beliefs:
                                pgco_logger.info(f"    {belief_type}: {belief['value']} (confidence: {belief['confidence']:.2f})")
                
            except Exception as e:
                pgco_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            pgco_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        pgco_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        pgco_logger.info("="*80)
        pgco_logger.info(prompt)
        pgco_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        pgco_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        pgco_logger.info(f"  Action: {decision['action']}")
        pgco_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        pgco_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log market context for analysis
        pgco_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        pgco_logger.info(f"  Current Job: {self.job}")
        pgco_logger.info(f"  Balance: ${self.balance}")
        pgco_logger.info(f"  Inventory: {self.inventory}")
        pgco_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        pgco_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        pgco_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        pgco_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            pgco_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            pgco_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            pgco_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            pgco_logger.info(f"📦 PGCO Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            print(f"🧠 PGCO Trader {self.tid} BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                pgco_logger.info(f"{emoji} PGCO Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                print(f"{emoji} PGCO Trader {self.tid} SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
            else:
                profit = 0
                pgco_logger.info(f"🔴 PGCO Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 PGCO Trader {self.tid} SOLD at ${transactionprice} | No purchase price recorded")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)

    def _generate_natural_language_insights(self):
        """
        Convert the belief graph JSON into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        # For Perfect Graph agents, get valuation data from belief edges
        def get_perfect_valuation(agent_id):
            """Get valuation from belief edges (perfect data)"""
            for edge in self.belief_graph.edges.values():
                if edge.target_node == agent_id and edge.belief_type == "valuation":
                    return edge.value, edge.confidence
            return None, 0.0
        
        # Analyze trading patterns
        insights.append("=== MARKET BEHAVIOR ANALYSIS ===")
        
        # Group agents by strategy
        aggressive_agents = []
        passive_agents = []
        neutral_agents = []
        
        for agent_id, node in agents.items():
            if node.strategy_type == "aggressive":
                aggressive_agents.append((agent_id, node))
            elif node.strategy_type == "passive":
                passive_agents.append((agent_id, node))
            else:
                neutral_agents.append((agent_id, node))
        
        # Strategy insights
        if aggressive_agents:
            insights.append(f"AGGRESSIVE TRADERS ({len(aggressive_agents)}): These agents tend to pay premium prices or accept lower selling prices to execute trades quickly.")
            for agent_id, node in aggressive_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        if passive_agents:
            insights.append(f"PASSIVE TRADERS ({len(passive_agents)}): These agents wait for better prices and are more patient.")
            for agent_id, node in passive_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        if neutral_agents:
            insights.append(f"NEUTRAL TRADERS ({len(neutral_agents)}): These agents trade at market prices without strong urgency.")
            for agent_id, node in neutral_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        # Price analysis
        insights.append("\n=== VALUATION PATTERNS ===")
        valuations = [(node.inferred_valuation, agent_id, node.strategy_type) 
                     for agent_id, node in agents.items() 
                     if node.inferred_valuation is not None]
        
        if valuations:
            valuations.sort(reverse=True)  # Highest to lowest
            highest_val = valuations[0]
            lowest_val = valuations[-1]
            
            insights.append(f"Highest valuation: {highest_val[1]} values at ${highest_val[0]} (strategy: {highest_val[2]})")
            insights.append(f"Lowest valuation: {lowest_val[1]} values at ${lowest_val[0]} (strategy: {lowest_val[2]})")
            
            avg_val = sum(v[0] for v in valuations) / len(valuations)
            insights.append(f"Average market valuation: ${avg_val:.1f}")
        
        # Temporal insights from event history
        if hasattr(self.belief_graph, 'event_history') and self.belief_graph.event_history:
            insights.append("\n=== RECENT TRADING SEQUENCE ===")
            recent_events = self.belief_graph.event_history[-5:]  # Last 5 events
            
            for i, event in enumerate(recent_events):
                if event.agent_id in agents:
                    agent_node = agents[event.agent_id]
                    insights.append(f"{i+1}. {event.agent_id} ({agent_node.strategy_type}) traded at ${event.price}")
        
        # Strategic recommendations
        insights.append("\n=== STRATEGIC INSIGHTS ===")
        
        if aggressive_agents and passive_agents:
            avg_aggressive_price = sum(node.last_trade_price for _, node in aggressive_agents) / len(aggressive_agents)
            avg_passive_price = sum(node.last_trade_price for _, node in passive_agents) / len(passive_agents)
            
            if avg_aggressive_price > avg_passive_price:
                insights.append(f"Aggressive traders are paying ${avg_aggressive_price - avg_passive_price:.1f} more on average than passive traders.")
                insights.append("This suggests there may be opportunities to be more patient and get better prices.")
            else:
                insights.append("Aggressive and passive traders are getting similar prices, suggesting a balanced market.")
        
        return "\n".join(insights)

class TraderPerfectGraphWithoutCoT(Trader):
    """
    LLM-based proprietary trader that uses an explicit belief graph for state management,
    utility inference, and decision making. This trader maintains a structured representation
    of other agents' behaviors and market state to make more informed trading decisions.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Belief Graph trader
        :param ttype: the trader type
        :param tid: the trader I.D.
        :param balance: starting balance
        :param params: parameters including API key and belief graph configuration
        :param time: current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Import belief graph components
        try:
            from agents.belief_graph import PerfectBeliefGraph, MarketEvent, EventType
            self.belief_graph = PerfectBeliefGraph(asset_id="BSE_ASSET", traders_dict=None)  # Will be set later
            self.MarketEvent = MarketEvent
            self.EventType = EventType
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
            self.MarketEvent = None
            self.EventType = None
        
        # LLM configuration (same as LLM trader for fair comparison)
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3  # Same as LLM trader
        self.max_tokens = 4000   # Same as LLM trader
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize the LLM
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            pgno_logger.info(f"Initialized PGNO (Perfect Graph + No CoT) trader {tid} with model {self.model_name}")
        else:
            pgno_logger.warning(f"No API key provided for PGNO (Perfect Graph + No CoT) trader {tid}")
            self.model = None
        
        # Trading state (same as LLM trader)
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None
        self.inventory = 0
        
        # Belief graph tracking (ONLY advantage over LLM trader)
        self.known_agents = set()
        self.last_market_update = 0.0
        self.belief_update_interval = 1.0  # Update beliefs every second
        
        # Trading history for context (same as LLM trader)
        self.trading_history = []
        self.max_history = 20  # Same as LLM trader
        
        # Performance tracking (same as LLM trader)
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Default trading parameters (same as LLM trader)
        self.n_past_trades = 5      # how many recent trades to analyze
        self.min_profit_margin = 5  # minimum profit we want when selling
        
        if params is not None:
            if 'n_past_trades' in params:
                self.n_past_trades = params['n_past_trades']
            if 'min_profit_margin' in params:
                self.min_profit_margin = params['min_profit_margin']
        
        # Debug mode
        self.debug_mode = False

    def set_traders_dict(self, traders_dict):
        """
        Set the traders dictionary for perfect belief graph access.
        This must be called after all traders are created.
        """
        if self.belief_graph is not None:
            self.belief_graph.traders_dict = traders_dict

    def _update_belief_graph_from_market(self, lob, time):
        """
        Update the belief graph with current market events
        """
        if not self.belief_graph:
            return
        
        # Add ourselves to the belief graph if not already present
        if self.tid not in self.known_agents:
            self.belief_graph.add_agent(self.tid)
            self.known_agents.add(self.tid)
        
        # Process recent market events from the tape
        if 'tape' in lob and lob['tape']:
            new_events_count = 0
            for event in lob['tape'][-10:]:  # Process last 10 events
                if event['time'] > self.last_market_update:
                    # Log what we're about to process for GraphVar1
                    event_type = event.get('type', 'Unknown')
                    event_price = event.get('price', 'N/A')
                    event_party = event.get('party1', event.get('agent', 'Unknown'))
                    gv1_logger.debug(f"[GraphVar1-PROCESSING] {self.tid}: Processing {event_type} event from {event_party} at price {event_price}, time={event['time']}")
                    self._process_market_event(event, time)
                    new_events_count += 1
            if new_events_count > 0:
                gv1_logger.info(f"[GraphVar1-UPDATE-SUMMARY] {self.tid}: Processed {new_events_count} new market events at time {time}")
                # Update to the timestamp of the last processed event, not current time
                if lob['tape']:
                    self.last_market_update = max(e['time'] for e in lob['tape'][-10:])
        else:
            gv1_logger.debug(f"[BG-UPDATE] {self.tid}: No tape data available")

    def _process_market_event(self, event, time):
        """
        Process a market event and update the belief graph
        """
        if not self.belief_graph:
            return
        
        # Determine event type
        if event['type'] == 'Trade':
            event_type = self.EventType.TRADE
        elif event['type'] == 'Bid':
            event_type = self.EventType.BID
        elif event['type'] == 'Ask':
            event_type = self.EventType.ASK
        else:
            return  # Skip other event types
        
        # Create market event for belief graph
        market_event = self.MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=event.get('time', time),
            agent_id=event.get('party1') or event.get('agent'),
            price=event.get('price'),
            quantity=event.get('qty', 1),
            counterparty_id=event.get('party2')
        )
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)

    def _get_belief_graph_context(self, lob, time):
        """
        Get the belief graph context for LLM decision making
        """
        if not self.belief_graph:
            return {"error": "Belief graph not available"}
        
        # Update belief graph with current market state
        self._update_belief_graph_from_market(lob, time)
        
        # Get current market state
        last_trade_price = None
        if 'tape' in lob and lob['tape']:
            try:
                last_event = lob['tape'][-1]
                if 'price' in last_event:
                    last_trade_price = last_event['price']
            except (IndexError, KeyError):
                pass
        
        current_market_state = {
            'best_bid': lob['bids']['best'] if lob['bids']['n'] > 0 else None,
            'best_ask': lob['asks']['best'] if lob['asks']['n'] > 0 else None,
            'last_trade': last_trade_price
        }
        
        # Query belief graph for decision context
        belief_context = self.belief_graph.query_action(self.tid, current_market_state)
        
        # Add natural language graph insights
        belief_context['natural_language_insights'] = self._generate_natural_language_insights()
        
        return belief_context

    def _format_belief_graph_prompt(self, belief_context, lob, time):
        """
        Format the belief graph data into a comprehensive prompt for the LLM
        """
        # Check if belief graph is available
        if isinstance(belief_context, dict) and 'error' in belief_context:
            # Fallback to basic prompt without belief graph
            return self._format_basic_prompt(lob, time)
        
        # Extract key information from belief context
        agents_info = belief_context.get('agents', {})
        strategic_insights = belief_context.get('strategic_insights', {})
        asset_state = belief_context.get('asset_state', {})
        
        # Format competitor analysis
        competitors_text = ""
        if strategic_insights.get('competitors'):
            competitors_text = "COMPETITOR ANALYSIS:\n"
            for comp in strategic_insights['competitors']:
                if comp['agent_id'] != self.tid:
                    valuation_str = f"${comp['valuation_estimate']:.1f}" if comp['valuation_estimate'] is not None else "Unknown"
                    competitors_text += f"- {comp['agent_id']}: Strategy={comp['strategy']}, "
                    competitors_text += f"Aggressiveness={comp['aggressiveness']:.2f}, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {comp['confidence']:.2f})\n"
        else:
            # Fallback: extract directly from belief data if strategic insights are empty
            competitors_text = "COMPETITOR ANALYSIS:\n"
            competitors_found = False
            if 'beliefs' in belief_context:
                agents_valuations = {}
                for belief in belief_context['beliefs']:
                    if (belief['belief_type'] == 'valuation' and 
                        belief['target_node'] != self.tid and 
                        belief['value'] is not None):
                        agent_id = belief['target_node']
                        agents_valuations[agent_id] = {
                            'valuation': belief['value'],
                            'confidence': belief['confidence']
                        }
                
                for agent_id, data in agents_valuations.items():
                    valuation_str = f"${data['valuation']:.1f}"
                    competitors_text += f"- {agent_id}: Strategy=unknown, "
                    competitors_text += f"Aggressiveness=0.00, "
                    competitors_text += f"Valuation≈{valuation_str} "
                    competitors_text += f"(confidence: {data['confidence']:.2f})\n"
                    competitors_found = True
            
            if not competitors_found:
                competitors_text += "- No competitor data available\n"
        
        # Format market opportunities
        opportunities_text = ""
        if strategic_insights.get('market_opportunities'):
            opportunities_text = "MARKET OPPORTUNITIES:\n"
            for opp in strategic_insights['market_opportunities']:
                opportunities_text += f"- {opp['description']}\n"
        
        # Format risk factors
        risks_text = ""
        if strategic_insights.get('risk_factors'):
            risks_text = "RISK FACTORS:\n"
            for risk in strategic_insights['risk_factors']:
                risks_text += f"- {risk['description']}\n"
        
        # Add natural language insights from belief graph
        natural_insights = ""
        if belief_context.get('natural_language_insights'):
            natural_insights = f"\nBELIEF GRAPH INSIGHTS:\n{belief_context['natural_language_insights']}\n"
        
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - consider how your current balance reflects your trading performance
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Successful prop traders typically aim for small, consistent profits rather than big gambles

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market conditions

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

TRADING PRINCIPLES TO CONSIDER:
- "Buy low, sell high" means buying below recent average prices when possible
- Risk management: avoid spending your entire balance on one trade
- Learn from history: if recent trades lost money, consider what went wrong
- Liquidity: sometimes waiting for better prices is smarter than forcing trades
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- You maintain a structured model of other agents' strategies and valuations
- Use this information to anticipate market movements and competitor actions
- Consider which agents are most likely to undercut your bids or accept your offers

STRATEGIC CONSIDERATIONS:
- Analyze competitor aggressiveness to predict price movements
- Use valuation estimates to identify mispriced opportunities
- Consider market opportunities and risk factors in your decision
- Balance immediate execution vs. waiting for better prices

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            prompt = f"""You are a sophisticated proprietary trader using a belief graph to model other agents' behaviors and market dynamics.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

{competitors_text}
{opportunities_text}
{risks_text}
{natural_insights}
YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- Recent average price: {avg_price_str}
- You started with $500 - your current balance shows your trading track record
- You MUST make a profit. Your goal is to end with MORE money than you started with.
- Every sale is an opportunity to learn and improve your strategy

TRADER ANALYSIS:
- Pay attention to which other traders are active and their average prices
- Analyze their trading patterns and use this information to inform your decisions
- Consider what their activity might indicate about market demand

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

TRADING PRINCIPLES TO CONSIDER:
- Profit target: what's a reasonable profit margin for this trade?
- Risk management: sometimes taking a small loss prevents a bigger loss
- Market conditions: is the market trending up or down?
- Patience vs urgency: waiting might get a better price, or price might fall further
- Learning: what does this trade teach you about timing and pricing?
- Trader behavior: analyze other traders' activity and draw your own conclusions

BELIEF GRAPH INSIGHTS:
- Use competitor analysis to predict who might buy at what price
- Consider agent aggressiveness to time your sale optimally
- Use valuation estimates to identify the best selling opportunities

STRATEGIC CONSIDERATIONS:
- Analyze which competitors are most likely to accept your ask price
- Consider market opportunities and risk factors
- Balance profit maximization vs. execution certainty
- Use belief graph insights to optimize timing and pricing

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()
        
        return prompt

    def _format_basic_prompt(self, lob, time):
        """
        Format a basic prompt when belief graph is not available
        """
        # Current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        # Recent price history
        recent_prices = []
        if 'tape' in lob and lob['tape']:
            for event in lob['tape'][-5:]:
                if event['type'] == 'Trade':
                    recent_prices.append(event['price'])
        
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"{avg_price:.1f}" if avg_price is not None else "N/A"
        
        if self.job == 'Buy':
            return f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT SITUATION: You have ${self.balance} and are looking to BUY a unit. You have NO INVENTORY.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        elif self.job == 'Sell':
            return f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

MARKET STATE:
- Best Bid: ${best_bid}
- Best Ask: ${best_ask}
- Spread: ${spread}
- Recent Prices: {recent_prices}
- Average Recent Price: {avg_price_str}

YOUR PERFORMANCE:
- Total Profit: ${self.total_profit:.2f}
- Successful Trades: {self.successful_trades}
- Failed Trades: {self.failed_trades}
- Current Balance: ${self.balance}

TRADE ANALYSIS:
- Purchase Price: ${self.last_purchase_price}
- Break-even: ${self.last_purchase_price}
- Minimum Profit Target: ${self.last_purchase_price + 2}

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price
"WAIT" - to wait for better conditions

No explanation needed."""
        
        else:
            return self._fallback_decision()

    def _get_llm_trading_decision(self, prompt):
        """
        Get trading decision from LLM using belief graph context
        """
        if not self.model:
            pgno_logger.warning(f"[BG-NO-MODEL] {self.tid}: No LLM model available, using fallback")
            return self._fallback_decision()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            response_text = response.text.strip()
            
            # Log the raw LLM response
            pgno_logger.info(f"[BG-LLM-RESPONSE] {self.tid}: === RAW LLM RESPONSE ===")
            pgno_logger.info("-"*50)
            pgno_logger.info(response_text)
            pgno_logger.info("-"*50)
            
            parsed_decision = self._parse_llm_response(response_text)
            
            # Log the parsed decision
            pgno_logger.info(f"[BG-PARSED-DECISION] {self.tid}: Parsed decision:")
            pgno_logger.info(f"  Action: {parsed_decision['action']}")
            pgno_logger.info(f"  Price: {parsed_decision.get('price', 'N/A')}")
            pgno_logger.info(f"  Full Reasoning: {parsed_decision.get('reasoning', 'N/A')}")
            
            return parsed_decision
            
        except Exception as e:
            pgno_logger.error(f"[BG-LLM-ERROR] {self.tid}: LLM API error: {e}")
            if self.debug_mode:
                print(f"LLM API error for Belief Graph trader {self.tid}: {e}")
            return self._fallback_decision()

    def _parse_llm_response(self, response_text):
        """
        Parse LLM response into actionable decision
        """
        response_upper = response_text.upper()
        
        import re
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match and self.job == 'Buy':
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match and self.job == 'Sell':
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text
            }
        
        # Default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text
        }

    def _fallback_decision(self):
        """
        Simple fallback decision if LLM is unavailable
        """
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }

    def getorder(self, time, countdown, lob):
        """
        Return this trader's order when polled in the main market session loop
        """
        if countdown < 0:
            sys.exit('Negative countdown')

        if len(self.orders) < 1 or time < 0.1 * 60:
            order = None
        else:
            # We have an order, execute it
            quoteprice = self.orders[0].price
            order = Order(self.tid, self.orders[0].otype, quoteprice, 
                         self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using belief graph
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        pgno_logger.debug(f"[BG-RESPOND] {self.tid}: respond() called at time {time:.1f}")
        
        # Get belief graph context and LLM decision
        belief_context = self._get_belief_graph_context(lob, time)
        
        # COMPREHENSIVE GRAPH VISUALIZATION LOGGING
        if self.belief_graph:
            pgno_logger.info(f"[BG-GRAPH-VIZ] {self.tid}: === COMPLETE BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                # Log the complete graph as JSON
                graph_json = self.belief_graph.to_json()
                pgno_logger.info(f"[BG-GRAPH-JSON] {self.tid}:")
                pgno_logger.info("="*80)
                pgno_logger.info(graph_json)
                pgno_logger.info("="*80)
                
                # Also log a human-readable summary
                market_summary = self.belief_graph.get_market_summary()
                pgno_logger.info(f"[BG-GRAPH-SUMMARY] {self.tid}: Market Summary:")
                pgno_logger.info(f"  Active Agents: {market_summary['active_agents']}")
                pgno_logger.info(f"  Total Beliefs: {market_summary['total_beliefs']}")
                pgno_logger.info(f"  Recent Events: {market_summary['recent_events']}")
                
                # Log beliefs about each agent
                for node_id, node in self.belief_graph.nodes.items():
                    if hasattr(node, 'agent_id') and node_id != self.tid:  # Skip self
                        agent_beliefs = self.belief_graph.get_agent_beliefs(node_id)
                        pgno_logger.info(f"[BG-AGENT-BELIEFS] {self.tid}: Beliefs about {node_id}:")
                        for belief_type, beliefs in agent_beliefs.items():
                            for belief in beliefs:
                                pgno_logger.info(f"    {belief_type}: {belief['value']} (confidence: {belief['confidence']:.2f})")
                
            except Exception as e:
                pgno_logger.error(f"[BG-GRAPH-ERROR] {self.tid}: Failed to log graph visualization: {e}")
        else:
            pgno_logger.warning(f"[BG-NO-GRAPH] {self.tid}: Belief graph not available for visualization")
        
        # Format the prompt with belief graph context
        prompt = self._format_belief_graph_prompt(belief_context, lob, time)
        
        # COMPREHENSIVE PROMPT LOGGING
        pgno_logger.info(f"[BG-PROMPT] {self.tid}: === COMPLETE LLM PROMPT AT TIME {time:.1f} ===")
        pgno_logger.info("="*80)
        pgno_logger.info(prompt)
        pgno_logger.info("="*80)
        
        # Get LLM decision
        decision = self._get_llm_trading_decision(prompt)
        
        # Log the LLM response
        pgno_logger.info(f"[BG-DECISION] {self.tid}: === LLM DECISION AT TIME {time:.1f} ===")
        pgno_logger.info(f"  Action: {decision['action']}")
        pgno_logger.info(f"  Price: {decision.get('price', 'N/A')}")
        pgno_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Log market context for analysis
        pgno_logger.info(f"[BG-CONTEXT] {self.tid}: Market context at decision time:")
        pgno_logger.info(f"  Current Job: {self.job}")
        pgno_logger.info(f"  Balance: ${self.balance}")
        pgno_logger.info(f"  Inventory: {self.inventory}")
        pgno_logger.info(f"  Best Bid: {lob['bids']['best'] if lob['bids']['n'] > 0 else 'None'}")
        pgno_logger.info(f"  Best Ask: {lob['asks']['best'] if lob['asks']['n'] > 0 else 'None'}")
        pgno_logger.info(f"  Last Purchase Price: {self.last_purchase_price}")
        
        # Log the decision
        self.trading_history.append({
            'time': time,
            'decision': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]

        # Act on the decision
        pgno_logger.debug(f"[BG-EXECUTE] {self.tid}: Executing decision - job={self.job}, decision={decision['action']}")
        if decision['action'] == 'BUY' and self.job == 'Buy':
            pgno_logger.debug(f"[BG-BUY-EXECUTE] {self.tid}: Executing BUY decision at price {decision.get('price', 'N/A')}")
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.job == 'Sell':
            pgno_logger.debug(f"[BG-SELL-EXECUTE] {self.tid}: Executing SELL decision at price {decision.get('price', 'N/A')}")
            self._execute_sell_decision(decision, lob, time)
        else:
            pgno_logger.debug(f"[BG-NO-ACTION] {self.tid}: No action taken - job={self.job}, decision={decision['action']}")

    def _execute_buy_decision(self, decision, lob, time):
        """
        Execute a buy decision with LLM-specified price
        """
        if lob['asks']['n'] == 0:
            return  # No asks available

        buy_price = decision['price']
        if buy_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            order = Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
        elif self.debug_mode:
            print(f"Warning: {self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """
        Execute a sell decision with LLM-specified price
        """
        if lob['bids']['n'] == 0:
            return  # No bids available

        sell_price = decision['price']
        if sell_price is None:
            if self.debug_mode:
                print(f"Warning: {self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        order = Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader's records after a successful trade
        """
        # Standard bookkeeping
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        transactionprice = trade['price']
        
        if self.orders[0].otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            self.job = 'Sell'  # Switch to selling mode
            
            pgno_logger.info(f"📦 PGNO Trader BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            print(f"🧠 PGNO Trader {self.tid} BOUGHT at ${transactionprice} | Balance: ${self.balance}")
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'BOUGHT',
                'price': transactionprice,
                'new_balance': self.balance,
                'new_job': self.job
            })
            
        elif self.orders[0].otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if profit >= 0:
                    self.successful_trades += 1
                    emoji = "🟢"
                else:
                    self.failed_trades += 1
                    emoji = "🔴"
                pgno_logger.info(f"{emoji} PGNO Trader SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
                print(f"{emoji} PGNO Trader {self.tid} SOLD at ${transactionprice} | Profit: ${profit} | Total Profit: ${self.total_profit:.2f}")
            else:
                profit = 0
                pgno_logger.info(f"🔴 PGNO Trader SOLD at ${transactionprice} | No purchase price recorded")
                print(f"🔴 PGNO Trader {self.tid} SOLD at ${transactionprice} | No purchase price recorded")
            
            self.inventory = 0
            self.last_purchase_price = None
            self.job = 'Buy'  # Switch back to buying mode
            
            # Log the state change
            self.trading_history.append({
                'time': time,
                'event': 'SOLD', 
                'price': transactionprice,
                'profit': profit,
                'new_balance': self.balance,
                'new_job': self.job
            })

        # Update trade count and profit per time
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime) if time > self.birthtime else 0

        # Clear the executed order
        self.del_order(order)

    def _generate_natural_language_insights(self):
        """
        Convert the belief graph JSON into natural language insights for the LLM
        """
        if not self.belief_graph:
            return "No belief graph available."
        
        insights = []
        
        # Get all agent nodes (excluding asset node)
        agents = {k: v for k, v in self.belief_graph.nodes.items() 
                 if hasattr(v, 'agent_id') and v.agent_id != self.belief_graph.asset_id}
        
        if not agents:
            return "No other agents observed yet."
        
        # For Perfect Graph agents, get valuation data from belief edges
        def get_perfect_valuation(agent_id):
            """Get valuation from belief edges (perfect data)"""
            for edge in self.belief_graph.edges.values():
                if edge.target_node == agent_id and edge.belief_type == "valuation":
                    return edge.value, edge.confidence
            return None, 0.0
        
        # Analyze trading patterns
        insights.append("=== MARKET BEHAVIOR ANALYSIS ===")
        
        # Group agents by strategy
        aggressive_agents = []
        passive_agents = []
        neutral_agents = []
        
        for agent_id, node in agents.items():
            if node.strategy_type == "aggressive":
                aggressive_agents.append((agent_id, node))
            elif node.strategy_type == "passive":
                passive_agents.append((agent_id, node))
            else:
                neutral_agents.append((agent_id, node))
        
        # Strategy insights
        if aggressive_agents:
            insights.append(f"AGGRESSIVE TRADERS ({len(aggressive_agents)}): These agents tend to pay premium prices or accept lower selling prices to execute trades quickly.")
            for agent_id, node in aggressive_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        if passive_agents:
            insights.append(f"PASSIVE TRADERS ({len(passive_agents)}): These agents wait for better prices and are more patient.")
            for agent_id, node in passive_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        if neutral_agents:
            insights.append(f"NEUTRAL TRADERS ({len(neutral_agents)}): These agents trade at market prices without strong urgency.")
            for agent_id, node in neutral_agents:
                perfect_val, perfect_conf = get_perfect_valuation(agent_id)
                valuation_str = f"${perfect_val:.1f}" if perfect_val is not None else "Unknown"
                insights.append(f"  • {agent_id}: Last traded at ${node.last_trade_price}, values asset around {valuation_str} (confidence: {perfect_conf:.0%})")
        
        # Price analysis
        insights.append("\n=== VALUATION PATTERNS ===")
        valuations = [(node.inferred_valuation, agent_id, node.strategy_type) 
                     for agent_id, node in agents.items() 
                     if node.inferred_valuation is not None]
        
        if valuations:
            valuations.sort(reverse=True)  # Highest to lowest
            highest_val = valuations[0]
            lowest_val = valuations[-1]
            
            insights.append(f"Highest valuation: {highest_val[1]} values at ${highest_val[0]} (strategy: {highest_val[2]})")
            insights.append(f"Lowest valuation: {lowest_val[1]} values at ${lowest_val[0]} (strategy: {lowest_val[2]})")
            
            avg_val = sum(v[0] for v in valuations) / len(valuations)
            insights.append(f"Average market valuation: ${avg_val:.1f}")
        
        # Temporal insights from event history
        if hasattr(self.belief_graph, 'event_history') and self.belief_graph.event_history:
            insights.append("\n=== RECENT TRADING SEQUENCE ===")
            recent_events = self.belief_graph.event_history[-5:]  # Last 5 events
            
            for i, event in enumerate(recent_events):
                if event.agent_id in agents:
                    agent_node = agents[event.agent_id]
                    insights.append(f"{i+1}. {event.agent_id} ({agent_node.strategy_type}) traded at ${event.price}")
        
        # Strategic recommendations
        insights.append("\n=== STRATEGIC INSIGHTS ===")
        
        if aggressive_agents and passive_agents:
            avg_aggressive_price = sum(node.last_trade_price for _, node in aggressive_agents) / len(aggressive_agents)
            avg_passive_price = sum(node.last_trade_price for _, node in passive_agents) / len(passive_agents)
            
            if avg_aggressive_price > avg_passive_price:
                insights.append(f"Aggressive traders are paying ${avg_aggressive_price - avg_passive_price:.1f} more on average than passive traders.")
                insights.append("This suggests there may be opportunities to be more patient and get better prices.")
            else:
                insights.append("Aggressive and passive traders are getting similar prices, suggesting a balanced market.")
        
        return "\n".join(insights)


# #########################---Below lies the experiment/test-rig---##################


class TraderAdaptive(Trader):
    """
    LLM-based trader that can design and adapt its own trading attributes.
    
    This trader integrates with the belief graph and uses its attributes to influence
    trading decisions and belief formation.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the adaptive trader
        
        Args:
            ttype: Trader type identifier
            tid: Trader ID
            balance: Starting balance
            params: Trader parameters including API key and attribute settings
            time: Current time
        """
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Initialize attribute system
        self.attribute_manager = None
        self.attributes_initialized = False
        
        # LLM configuration
        self.api_key = None
        self.model_name = 'gemini-2.0-flash-lite'
        self.temperature = 0.3
        self.max_tokens = 500
        
        # Parse LLM parameters if provided
        if params is not None:
            if 'api_key' in params:
                self.api_key = params['api_key']
            if 'model_name' in params:
                self.model_name = params['model_name']
            if 'temperature' in params:
                self.temperature = params['temperature']
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize LLM if API key is available
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized Adaptive Trader {tid} with model {self.model_name}")
        else:
            print(f"Warning: No API key provided for Adaptive Trader {tid}")
            self.model = None
        
        # Belief graph integration
        try:
            self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
            self.belief_graph.add_agent(tid)
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
        
        # Trading state
        self.job = 'Buy'  # Buy or Sell mode
        self.last_purchase_price = None
        self.inventory = 0
        self.n_trades = 0
        
        # Performance tracking
        self.starting_balance = balance
        self.total_profit = 0.0
        self.trading_history = []
        
        # Attribute adaptation settings
        self.adaptation_enabled = True
        self.adaptation_interval = 10  # Every 10 trades
        self.last_adaptation_check = 0
        
        # Market context tracking
        self.market_context = {
            'volatility': 0.0,
            'trend': 'unknown',
            'competition': 'unknown',
            'liquidity': 'unknown'
        }
        
        # Initialize attributes if LLM is available
        if self.model:
            self.initialize_attributes()
        else:
            # Fallback: use balanced strategy if no LLM available
            self.initialize_attributes_fallback()
    
    def initialize_attributes(self):
        """Initialize the agent's attributes using LLM decision making"""
        if self.attributes_initialized:
            return
        
        # Update market context
        self._update_market_context()
        
        # Create initial design prompt
        available_strategies = ["random", "conservative", "aggressive", "balanced", "momentum", "mean_reversion"]
        
        prompt = f"""You are a trading agent designing your own trading personality for a financial market.

CURRENT MARKET CONDITIONS:
- Market volatility: {self.market_context.get('volatility', 'Unknown')}
- Recent price trend: {self.market_context.get('trend', 'Unknown')}
- Competition level: {self.market_context.get('competition', 'Unknown')}
- Available liquidity: {self.market_context.get('liquidity', 'Unknown')}

AVAILABLE DESIGN STRATEGIES:
- random
- conservative
- aggressive
- balanced
- momentum
- mean_reversion

YOUR MISSION:
Design your trading personality by choosing one of the available strategies. Each strategy creates a different combination of trading attributes that will define how you behave in the market.

RESPOND WITH ONLY:
"DESIGN: [strategy_name]"

No explanation needed. Choose the strategy that best fits your understanding of the current market conditions and your trading philosophy.

Examples:
- "DESIGN: conservative"
- "DESIGN: aggressive"
- "DESIGN: balanced"
- "DESIGN: momentum"
- "DESIGN: mean_reversion"
- "DESIGN: random"
"""
        
        try:
            # Get LLM response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Parse response and initialize attributes
            strategy = self._parse_design_response(response.text)
            self._initialize_attributes_from_strategy(strategy)
            self.attributes_initialized = True
            
            print(f"Adaptive Trader {self.tid} designed attributes using strategy: {strategy}")
            
        except Exception as e:
            print(f"Error in attribute design for trader {self.tid}: {e}")
            # Fallback to balanced strategy
            self._initialize_attributes_from_strategy("balanced")
            self.attributes_initialized = True
    
    def initialize_attributes_fallback(self):
        """Initialize attributes using fallback strategy when no LLM is available"""
        self._initialize_attributes_from_strategy("balanced")
        self.attributes_initialized = True
        print(f"Adaptive Trader {self.tid} using fallback balanced strategy")
    
    def _parse_design_response(self, response):
        """Parse the design strategy from LLM response"""
        response_upper = response.upper().strip()
        
        # Look for "DESIGN: [strategy]" pattern
        if "DESIGN:" in response_upper:
            strategy = response_upper.split("DESIGN:")[1].strip()
            return strategy.lower()
        
        # Fallback: look for strategy names in the response
        strategies = ["random", "conservative", "aggressive", "balanced", "momentum", "mean_reversion"]
        for strategy in strategies:
            if strategy.upper() in response_upper:
                return strategy
        
        # Default to balanced if no clear strategy found
        return "balanced"
    
    def _initialize_attributes_from_strategy(self, strategy):
        """Initialize attributes based on the chosen strategy"""
        if strategy == "random":
            self.attributes = {
                'aggressiveness': random.uniform(0.0, 1.0),
                'risk_tolerance': random.uniform(0.0, 1.0),
                'patience': random.uniform(0.0, 1.0),
                'adaptability': random.uniform(0.0, 1.0),
                'momentum_following': random.uniform(0.0, 1.0),
                'mean_reversion': random.uniform(0.0, 1.0)
            }
        elif strategy == "conservative":
            self.attributes = {
                'aggressiveness': random.uniform(0.0, 0.3),
                'risk_tolerance': random.uniform(0.0, 0.3),
                'patience': random.uniform(0.7, 1.0),
                'adaptability': random.uniform(0.3, 0.6),
                'momentum_following': random.uniform(0.2, 0.5),
                'mean_reversion': random.uniform(0.6, 1.0)
            }
        elif strategy == "aggressive":
            self.attributes = {
                'aggressiveness': random.uniform(0.7, 1.0),
                'risk_tolerance': random.uniform(0.7, 1.0),
                'patience': random.uniform(0.0, 0.3),
                'adaptability': random.uniform(0.6, 1.0),
                'momentum_following': random.uniform(0.6, 1.0),
                'mean_reversion': random.uniform(0.0, 0.3)
            }
        elif strategy == "balanced":
            self.attributes = {
                'aggressiveness': random.uniform(0.4, 0.6),
                'risk_tolerance': random.uniform(0.4, 0.6),
                'patience': random.uniform(0.4, 0.6),
                'adaptability': random.uniform(0.4, 0.6),
                'momentum_following': random.uniform(0.4, 0.6),
                'mean_reversion': random.uniform(0.4, 0.6)
            }
        elif strategy == "momentum":
            self.attributes = {
                'aggressiveness': random.uniform(0.5, 0.8),
                'risk_tolerance': random.uniform(0.5, 0.8),
                'patience': random.uniform(0.2, 0.5),
                'adaptability': random.uniform(0.6, 1.0),
                'momentum_following': random.uniform(0.8, 1.0),
                'mean_reversion': random.uniform(0.0, 0.2)
            }
        elif strategy == "mean_reversion":
            self.attributes = {
                'aggressiveness': random.uniform(0.3, 0.6),
                'risk_tolerance': random.uniform(0.4, 0.7),
                'patience': random.uniform(0.6, 1.0),
                'adaptability': random.uniform(0.3, 0.6),
                'momentum_following': random.uniform(0.0, 0.3),
                'mean_reversion': random.uniform(0.8, 1.0)
            }
        else:
            # Default to balanced
            self.attributes = {
                'aggressiveness': 0.5,
                'risk_tolerance': 0.5,
                'patience': 0.5,
                'adaptability': 0.5,
                'momentum_following': 0.5,
                'mean_reversion': 0.5
            }
        
        # Add design strategy to attributes for belief graph
        self.attributes['design_strategy'] = strategy
        
        # Update belief graph with the designed attributes
        if self.belief_graph:
            try:
                self.belief_graph.update_agent_attributes(self.tid, self.attributes)
                print(f"Trader {self.tid} updated belief graph with designed attributes: {strategy}")
            except Exception as e:
                print(f"Error updating belief graph for trader {self.tid}: {e}")
    
    def _update_market_context(self):
        """Update market context based on current conditions"""
        # This would be populated with real market data during trading
        self.market_context.update({
            'volatility': 'Medium',
            'trend': 'Stable',
            'competition': 'Moderate',
            'liquidity': 'High'
        })
    
    def should_check_adaptation(self):
        """Check if it's time to consider attribute adaptation"""
        if not self.adaptation_enabled:
            return False
        
        return self.n_trades >= self.last_adaptation_check + self.adaptation_interval
    
    def check_and_adapt_attributes(self):
        """Check if attributes should be adapted and perform adaptation if needed"""
        if not self.should_check_adaptation():
            return
        
        if not self.model or not self.attributes_initialized:
            return
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Check if adaptation is needed
        if self._should_adapt(performance_metrics):
            strategy = self._determine_adaptation_strategy(performance_metrics)
            self._apply_adaptation(strategy, performance_metrics)
            print(f"Trader {self.tid} adapted attributes to {strategy}")
        
        # Update adaptation check timestamp
        self.last_adaptation_check = self.n_trades
    
    def _should_adapt(self, performance_metrics):
        """Determine if attributes should be adapted"""
        # Adapt if performance is poor
        if performance_metrics.get('profit', 0) < -50:  # Lost more than $50
            return True
        
        # Adapt if market conditions have changed significantly
        if performance_metrics.get('market_volatility', 0) > 0.8:
            return True
        
        # Adapt if other agents are outperforming significantly
        if performance_metrics.get('relative_performance', 0) < -0.2:
            return True
        
        return False
    
    def _determine_adaptation_strategy(self, performance):
        """Determine which adaptation strategy to use"""
        if performance.get('profit', 0) < -100:
            return "conservative"  # Big losses -> become more conservative
        elif performance.get('market_volatility', 0) > 0.8:
            return "balanced"      # High volatility -> become more balanced
        elif performance.get('relative_performance', 0) < -0.3:
            return "aggressive"    # Underperforming -> become more aggressive
        else:
            return "balanced"      # Default to balanced
    
    def _apply_adaptation(self, strategy, performance_metrics):
        """Apply adaptation strategy to current attributes"""
        # Get new base attributes
        new_attributes = self._get_strategy_attributes(strategy)
        
        # Blend with current attributes based on adaptability
        blend_factor = self.attributes.get('adaptability', 0.5)
        
        for attr in self.attributes:
            if attr in new_attributes:
                self.attributes[attr] = self._blend_attribute(
                    self.attributes[attr],
                    new_attributes[attr],
                    blend_factor
                )
        
        # Update design strategy in attributes
        self.attributes['design_strategy'] = f"{strategy}_adapted"
        
        # Update belief graph with adapted attributes
        if self.belief_graph:
            try:
                self.belief_graph.update_agent_attributes(self.tid, self.attributes)
                print(f"Trader {self.tid} updated belief graph with adapted attributes: {strategy}")
            except Exception as e:
                print(f"Error updating belief graph for trader {self.tid}: {e}")
    
    def _get_strategy_attributes(self, strategy):
        """Get base attributes for a strategy"""
        if strategy == "conservative":
            return {
                'aggressiveness': random.uniform(0.0, 0.3),
                'risk_tolerance': random.uniform(0.0, 0.3),
                'patience': random.uniform(0.7, 1.0),
                'adaptability': random.uniform(0.3, 0.6),
                'momentum_following': random.uniform(0.2, 0.5),
                'mean_reversion': random.uniform(0.6, 1.0)
            }
        elif strategy == "aggressive":
            return {
                'aggressiveness': random.uniform(0.7, 1.0),
                'risk_tolerance': random.uniform(0.7, 1.0),
                'patience': random.uniform(0.0, 0.3),
                'adaptability': random.uniform(0.6, 1.0),
                'momentum_following': random.uniform(0.6, 1.0),
                'mean_reversion': random.uniform(0.0, 0.3)
            }
        elif strategy == "balanced":
            return {
                'aggressiveness': random.uniform(0.4, 0.6),
                'risk_tolerance': random.uniform(0.4, 0.6),
                'patience': random.uniform(0.4, 0.6),
                'adaptability': random.uniform(0.4, 0.6),
                'momentum_following': random.uniform(0.4, 0.6),
                'mean_reversion': random.uniform(0.4, 0.6)
            }
        else:
            return self._get_strategy_attributes("balanced")
    
    def _blend_attribute(self, current, target, blend_factor):
        """Blend current and target attribute values"""
        return current * (1 - blend_factor) + target * blend_factor
    
    def _calculate_performance_metrics(self):
        """Calculate current performance metrics for adaptation decisions"""
        current_profit = self.balance - self.starting_balance
        
        # Calculate market volatility (simplified)
        if len(self.trading_history) > 1:
            prices = [trade['price'] for trade in self.trading_history[-10:]]
            if len(prices) > 1:
                volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) / len(prices)
                volatility = min(volatility / 100.0, 1.0)  # Normalize to 0-1
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        # Calculate relative performance (simplified - would compare to other agents)
        relative_performance = 0.0  # Placeholder
        
        return {
            'profit': current_profit,
            'market_volatility': volatility,
            'relative_performance': relative_performance,
            'trade_count': self.n_trades,
            'timestamp': self.birthtime
        }
    
    def get_attributes_summary(self):
        """Get a summary of current attributes and adaptation history"""
        if not self.attributes_initialized:
            return {"error": "Attributes not initialized"}
        
        return {
            'current_attributes': self.attributes,
            'design_strategy': getattr(self, 'design_strategy', 'unknown'),
            'adaptation_enabled': self.adaptation_enabled,
            'performance_metrics': self._calculate_performance_metrics()
        }
    
    def respond(self, time, lob, trade, vrbs):
        """
        Main trading logic for the adaptive trader
        """
        # Check if adaptation is needed
        self.check_and_adapt_attributes()
        
        # Update belief graph if available
        if self.belief_graph:
            self._update_belief_graph(lob, time)
        
        # Update profit per time (required by BSE)
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        return None
    
    def getorder(self, time, countdown, lob):
        """
        Create this trader's order to be sent to the exchange.
        """
        # Check if adaptation is needed
        self.check_and_adapt_attributes()
        
        # Check if we have customer orders to work
        if len(self.orders) < 1:
            return None
        
        # Get current market state
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        
        if not best_bid or not best_ask:
            return None
        
        # Get the customer order we're working
        customer_order = self.orders[0]
        
        # Make trading decision based on attributes and customer order
        decision = self._make_trading_decision(lob, time, customer_order)
        
        return decision
    
    def _update_belief_graph(self, lob, time):
        """Update belief graph with current market state"""
        try:
            # Create market event for belief graph
            if lob['bids']['n'] > 0 and lob['asks']['n'] > 0:
                event = MarketEvent(
                    event_id=f"update_{time}",
                    event_type=EventType.BID,
                    timestamp=time,
                    agent_id=self.tid,
                    price=lob['bids']['best'],
                    quantity=1
                )
                self.belief_graph.update_beliefs(event)
        except Exception as e:
            print(f"Error updating belief graph: {e}")
    
    def _make_trading_decision(self, lob, time, customer_order):
        """Make trading decision influenced by agent attributes and customer order"""
        if not self.attributes_initialized:
            return None
        
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        
        if not best_bid or not best_ask:
            return None
        
        # Get current attributes
        aggressiveness = self.attributes.get('aggressiveness', 0.5)
        patience = self.attributes.get('patience', 0.5)
        momentum_following = self.attributes.get('momentum_following', 0.5)
        mean_reversion = self.attributes.get('mean_reversion', 0.5)
        
        # Calculate spread
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Get customer order details
        customer_type = customer_order.otype
        customer_price = customer_order.price
        
        # Decision logic based on attributes and customer order
        if customer_type == 'Bid':  # Customer wants to buy
            # Aggressive buyers bid higher
            bid_price = best_bid + (aggressiveness * spread * 0.1)
            
            # Patient buyers wait for better prices
            if patience > 0.7 and spread > 5:
                return None  # Wait for better spread
            
            # Momentum followers bid higher in rising markets
            if momentum_following > 0.7 and self._is_rising_market(lob):
                bid_price += spread * 0.2
            
            # Mean reversion traders bid lower in high markets
            if mean_reversion > 0.7 and self._is_high_market(lob):
                bid_price -= spread * 0.1
            
            # Ensure we don't exceed customer's limit price
            bid_price = min(bid_price, customer_price)
            
            return Order(self.tid, 'Bid', int(bid_price), 1, time, lob['QID'])
        
        else:  # customer_type == 'Ask' - Customer wants to sell
            # Aggressive sellers ask lower
            ask_price = best_ask - (aggressiveness * spread * 0.1)
            
            # Patient sellers wait for better prices
            if patience > 0.7 and spread > 5:
                return None  # Wait for better spread
            
            # Momentum followers ask higher in falling markets
            if momentum_following > 0.7 and self._is_falling_market(lob):
                ask_price += spread * 0.2
            
            # Mean reversion traders ask higher in low markets
            if mean_reversion > 0.7 and self._is_low_market(lob):
                ask_price += spread * 0.1
            
            # Ensure we don't go below customer's limit price
            ask_price = max(ask_price, customer_price)
            
            return Order(self.tid, 'Ask', int(ask_price), 1, time, lob['QID'])
    
    def _is_rising_market(self, lob):
        """Check if market is rising based on recent trades"""
        if len(lob['tape']) < 3:
            return False
        
        recent_trades = [t for t in lob['tape'][-3:] if t['type'] == 'Trade']
        if len(recent_trades) < 2:
            return False
        
        return recent_trades[-1]['price'] > recent_trades[-2]['price']
    
    def _is_falling_market(self, lob):
        """Check if market is falling based on recent trades"""
        if len(lob['tape']) < 3:
            return False
        
        recent_trades = [t for t in lob['tape'][-3:] if t['type'] == 'Trade']
        if len(recent_trades) < 2:
            return False
        
        return recent_trades[-1]['price'] < recent_trades[-2]['price']
    
    def _is_high_market(self, lob):
        """Check if market is at high levels"""
        if not lob['tape']:
            return False
        
        recent_trades = [t for t in lob['tape'][-5:] if t['type'] == 'Trade']
        if len(recent_trades) < 3:
            return False
        
        avg_price = sum(t['price'] for t in recent_trades) / len(recent_trades)
        return avg_price > 300  # Arbitrary threshold
    
    def _is_low_market(self, lob):
        """Check if market is at low levels"""
        if not lob['tape']:
            return False
        
        recent_trades = [t for t in lob['tape'][-5:] if t['type'] == 'Trade']
        if len(recent_trades) < 3:
            return False
        
        avg_price = sum(t['price'] for t in recent_trades) / len(recent_trades)
        return avg_price < 200  # Arbitrary threshold


def trade_stats(expid, traders, dumpfile, time, lob):
    """
    Dump CSV statistics on exchange data and trader population to file for later analysis.
    This makes no assumptions about the number of types of traders, or the number of traders of any one type
    -- allows either/both to change between successive calls, but that does make it inefficient as it has to
    re-analyse the entire set of traders on each call.
    :param expid: the experiment-I.D. character-string.
    :param traders: the list of traders in the market.
    :param dumpfile: the file that will be written to.
    :param time: the current time.
    :param lob: the current state of the LOB.
    :return: <nothing>
    """

    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    dumpfile.write('\n')


def populate_market(trdrs_spec, traders, shuffle, vrbs):
    """
    Create a bunch of traders from traders-specification.
    Optionally shuffles the pack of buyers and the pack of sellers.
    :param trdrs_spec: the specification of the population of traders.
    :param traders: the list into which the newly-created traders traders will be written, as a return parameter
    :param shuffle: whether to shuffle the ordering of buyers/sellers within the respective list.
    :param vrbs: verbosity Boolean: if True, print a running commentary; if False, stay silent.
    :return: tuple (n_buyers, n_sellers)
    """
    # trdrs_spec is a list of buyer-specs and a list of seller-specs
    # each spec is (<trader type>, <number of this type of trader>, optionally: <params for this type of trader>)

    def trader_type(robottype, name, parameters):
        """
        Create a newly instantiated trader of the designated type.
        :param robottype: the 'ticker-symbol' abbreviation indicating what type of trader to create.
        :param name: this trader's trader-I.D. character string.
        :param parameters: a list of parameter values for this trader-type.
        :return: a newly created trader of the designated type.
        """
        # UNIFIED CONFIGURATION APPROACH: Use the global configuration to create traders
        # Check if trader type is configured
        if robottype not in AVAILABLE_TRADER_TYPES:
            sys.exit(f'FATAL: Unknown trader type "{robottype}". Available types: {list(AVAILABLE_TRADER_TYPES.keys())}\n')
        
        trader_config = AVAILABLE_TRADER_TYPES[robottype]
        class_name = trader_config['class']
        balance_type = trader_config['balance_type']
        default_params = trader_config['params']
        
        # Set balance based on trader type
        if balance_type == 'prop':
            balance = 500  # Proprietary traders start with $500
        else:
            balance = 0.00  # Standard traders start with $0
            
        time0 = 0
        
        # Merge default parameters with provided parameters
        final_params = default_params.copy()
        if parameters:
            final_params.update(parameters)
        
        # Handle AgentFactory types
        if class_name == 'AgentFactory':
            from agents import AgentFactory
            return AgentFactory.create_agent(
                agent_type=robottype,
                tid=name,
                balance=balance,
                params=final_params,
                time=time0
            )

        # Dynamically create trader using globals() to get the class
        trader_class = globals().get(class_name)
        if trader_class is None:
            sys.exit(f'FATAL: Trader class "{class_name}" not found for type "{robottype}"\n')

        return trader_class(robottype, name, balance, final_params, time0)

    def shuffle_traders(ttype_char, n, trader_list):
        """
        Shuffles the trader-I.D. character strings of the traders in trader_list
        :param ttype_char: the lead character on the trader-I.D. strings (B for buyer, S for seller, etc)
        :param n: how many traders of this type
        :param trader_list: the list of traders in which the shuffling happens
        :return: <nothing>
        """
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            trader_list[t1name].tid = t2name
            trader_list[t2name].tid = t1name
            temp = traders[t1name]
            trader_list[t1name] = trader_list[t2name]
            trader_list[t2name] = temp

    def unpack_params(trader_params, mapping):
        """
        Unpack the parameters for those trader-types that have them
        :param trader_params: the paramaters being passed to this trader.
        :param mapping: Boolean flag: if True, enable fitness-landscape-mapping; otherwise do nothing for mapping.
        :return: the dictionary of parameters for this trader.
        """

        parameters = None

        if ttype == 'ZIPSH' or ttype == 'ZIP':
            # parameters matter...
            if mapping:
                parameters = 'landscape-mapper'
            elif trader_params is not None:
                parameters = trader_params.copy()
                # trader-type determines type of optimizer used
                if ttype == 'ZIPSH':
                    parameters['optimizer'] = 'ZIPSH'
                else:   # ttype=ZIP
                    parameters['optimizer'] = None
        if ttype == 'PRSH' or ttype == 'PRDE' or ttype == 'PRZI':
            # parameters matter...
            if mapping:
                parameters = 'landscape-mapper'
            elif trader_params is not None:
                # params determines type of optimizer used
                if ttype == 'PRSH':
                    parameters = {'optimizer': 'PRSH', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                elif ttype == 'PRDE':
                    parameters = {'optimizer': 'PRDE', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                else:   # ttype=PRZI
                    parameters = {'optimizer': None, 'k': 1,
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
            else:
                sys.exit('FAIL: PRZI/PRSH/PRDE trader needs one or more parameters to be specified')
                
        # for PT1/PT2 the parameters are optional...
        # ...and are unpacked in __init__, so here they're just passed straight on through
        if ttype == 'PT1':
            parameters = trader_params
        if ttype == 'PT2':
            parameters = trader_params

        return parameters

    landscape_mapping = False   # set to true when mapping fitness landscape (for PRSH etc).

    # the code that follows is a bit of a kludge, needs tidying up.
    n_buyers = 0
    for bs in trdrs_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            if len(bs) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(bs[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in trdrs_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            if len(ss) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(ss[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    n_proptraders = 0
    if 'proptraders' in trdrs_spec and len(trdrs_spec['proptraders']) > 0:
        for pts in trdrs_spec['proptraders']:
            ttype = pts[0]
            for pt in range(pts[1]):
                tname = 'P%02d' % n_proptraders  # proptrader i.d. string
                if len(pts) > 2:
                    # third part of the buyer-spec is params for this trader-type
                    params = unpack_params(pts[2], landscape_mapping)
                else:
                    params = unpack_params(None, landscape_mapping)
                traders[tname] = trader_type(ttype, tname, params)
                n_proptraders = n_proptraders + 1

    # NB markets with zero proptraders don't cause a fatal error

    if n_proptraders > 0 and shuffle:
        shuffle_traders('P', n_proptraders, traders)

    if vrbs:
        for t in range(n_buyers):
            tname = 'B%02d' % t
            print(traders[tname])
        for t in range(n_sellers):
            tname = 'S%02d' % t
            print(traders[tname])
        for t in range(n_proptraders):
            tname = 'P%02d' % t
            print(traders[tname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers, 'n_proptraders': n_proptraders}


def customer_orders(time, traders, trader_stats, orders_sched, pending, vrbs):
    """
    Generate a list of new customer-orders to be issued to the traders in the immediate/near future,
    and a list of any existing customer-orders that need to be cancelled because they are overridden by new ones.
    :param time: the current time.
    :param traders: the population of traders.
    :param trader_stats: summary statistics about the population of traders.
    :param orders_sched: the supply/demand schedule from which the orders will be generated...
            os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson';
            os['interval'] is number of seconds for a full cycle of replenishment;
            drip-poisson sequences will be normalised to ensure time of last replenishment <= interval.
            If a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
            then each time a price is generated one of the ranges is chosen equiprobably and the price is
            then generated uniform-randomly from that range.
            if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve.
            if len(range)==3, first two vals are min & max for linear sup/dem curves, and third value should be a
            callable function that generates a dynamic price offset; he offset value applies equally to the min & max,
            so gradient of linear sup/dem curves doesn't vary, but equilibrium price does.
            if len(range)==4, the third value is function that gives dynamic offset for schedule min, and 4th is a
            function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary dynamically
            along with the varying equilibrium price.
    :param pending: the list of currently pending future orders if this is empty, generates a new one).
    :param vrbs: verbosity Boolean: if True, print a running commentary; if False, stay silent.
    :return: [new_pending, cancellations]:
            new_pending is list of new orders to be issued;
            cancellations is list of previously-issued orders now cancelled.
    """

    def sysmin_check(price):
        """ if price is less than system minimum price, issue a warning and clip the price to the minimum"""
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        """ if price is greater than system maximum price, issue a warning and clip the price to the maximum"""
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, schedules, n, stepmode, orderissuetime):
        """
        Generate a price for an order, using the given supply/demand schedule, and specified step-mode.
        :param i: index of trader (position in list of traders).
        :param schedules: the supply/demand schedules.
        :param n: the number of traders that this schedule sup/dem is being applied to.
        :param stepmode: what type of steps to have between successive prices on the sup/dem schedule.
                stepmode=='fixed' => all steps are equal at one fixed size -- a "uniform-step" (see "jittered", below);
                stepmode=='jittered' => all steps are random, constrained to be within 2 uniform-steps of each other;
                stepmode=='random' => all steps are generated from a uniform distribution.
        :param orderissuetime: the time that this order will be issued at.
        :return: the price.
        """

        # does the first schedule range include optional dynamic offset function(s)?
        if len(schedules[0]) > 2:
            offsetfn = schedules[0][2]
            if callable(offsetfn[0]):
                # same offset for min and max
                offset_min = offsetfn[0](orderissuetime, *offsetfn[1])
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(schedules[0]) > 3:
                # if second offset function is specified, that applies only to the max value
                offsetfn = schedules[0][3]
                if callable(offsetfn):
                    # this function applies to max
                    offset_max = offsetfn(orderissuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(schedules[0][0], schedules[0][1]))
        pmax = sysmax_check(offset_max + max(schedules[0][0], schedules[0][1]))
        prange = pmax - pmin
        stepsize = prange / (n - 1)
        halfstep = round(stepsize / 2.0)

        if stepmode == 'fixed':
            order_price = pmin + int(i * stepsize)
        elif stepmode == 'jittered':
            order_price = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif stepmode == 'random':
            if len(schedules) > 1:
                # more than one schedule: choose one equiprobably
                s = random.randint(0, len(schedules) - 1)
                pmin = sysmin_check(min(schedules[s][0], schedules[s][1]))
                pmax = sysmax_check(max(schedules[s][0], schedules[s][1]))
            order_price = random.randint(int(pmin), int(pmax))
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        order_price = sysmin_check(sysmax_check(order_price))
        return order_price

    def getissuetimes(n_traders, timemode, interval, shuffle, fittointerval):
        """
        Generate a list of issue/arrival times for a set of future customer-orders, over a specified time-interval.
        :param n_traders: how many traders need issue times (i.e., the number of customer orders to be generated)
        :param timemode: character-string specifying the temporal spacing of orders:
                timemode=='periodic'=> orders issued to all traders at the same instant in time, every time-interval;
                timemode=='drip-fixed'=> order interarrival time is exactly one timestep, for all orders;
                timemode=='drip-jitter'=> order interarrival time is (1+r)*timestep, r=U[0,timestep], for all orders;
                timemode=='drip-poisson'=> order interarrival time is a Poisson random process, for all orders.
        :param interval: the time-interval between successive order issuals/arrivals.
        :param shuffle: if True then shuffle the arrival times, randomising the sequence in which traders get orders.
        :param fittointerval: if True then final order arrives at exactly t+interval; else may be slightly later.
        :return: the list of issue times.
        """
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetime()')
        elif n_traders == 1:
            tstep = interval
        else:
            tstep = interval / (n_traders - 1)
        arrtime = 0
        issue_times = []
        for trdr in range(n_traders):
            if timemode == 'periodic':
                arrtime = interval
            elif timemode == 'drip-fixed':
                arrtime = trdr * tstep
            elif timemode == 'drip-jitter':
                arrtime = trdr * tstep + tstep * random.random()
            elif timemode == 'drip-poisson':
                # poisson requires a bit of extra work
                interarrivaltime = random.expovariate(n_traders / interval)
                arrtime += interarrivaltime
            else:
                sys.exit('FAIL: unknown time-mode in getissuetimes()')
            issue_times.append(arrtime)
            # at this point, arrtime is the last arrival time

        if fittointerval and ((arrtime > interval) or (arrtime < interval)):
            # generated sum of interarrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for trdr in range(n_traders):
                issue_times[trdr] = interval * (issue_times[trdr] / arrtime)
        # optionally randomly shuffle the times
        if shuffle:
            for trdr in range(n_traders):
                i = (n_traders - 1) - trdr
                j = random.randint(0, i)
                tmp = issue_times[i]
                issue_times[i] = issue_times[j]
                issue_times[j] = tmp
        return issue_times

    def getschedmode(t_now, order_schedules):
        """
        return the step-mode for supply/demand schedule at the current time
        :param t_now: the current time
        :param order_schedules: dictionary/list of order schedules
        :return: schedrange = the price range for this schedule; mode= the stepmode for this schedule
        """
        got_one = False
        schedrange = None
        stepmode = None
        for schedule in order_schedules:
            if (schedule['from'] <= t_now) and (t_now < schedule['to']):
                # within the timezone for this schedule
                schedrange = schedule['ranges']
                stepmode = schedule['stepmode']
                got_one = True
                break  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in order_schedules=%s' % (t_now, order_schedules))
        return schedrange, stepmode

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issuetimes = getissuetimes(n_buyers, orders_sched['timemode'], orders_sched['interval'], shuffle_times, True)

        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, orders_sched['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)

        # supply side (sellers)
        issuetimes = getissuetimes(n_sellers, orders_sched['timemode'], orders_sched['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, orders_sched['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            # print('time %d sellerprice %d' % (time,orderprice))
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)
    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                tname = order.tid
                response = traders[tname].add_order(order, vrbs)
                if vrbs:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
                    if vrbs:
                        print('Cancellations: %s' % cancellations)
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations]


def calculate_prop_trader_net_worth(traders, lob=None):
    """Calculate net worth for all proprietary traders including inventory at current market value"""
    net_worths = {}
    
    for tid, trader in traders.items():
        # if trader.ttype in ['PT1', 'PT2', 'LLM', 'BG', 'BGNO', 'PGCO', 'PGNO', 'GV1', 'GV2']:
        if trader.ttype in PROP_TRADER_TYPES:
            net_worth = trader.balance
            
            # Check if trader is holding inventory
            inventory_value = 0
            if hasattr(trader, 'job') and trader.job == 'Sell':
                # Trader is in sell mode, so they have inventory
                if hasattr(trader, 'last_purchase_price') and trader.last_purchase_price is not None:
                    # Use current market value (best bid) instead of purchase price
                    if lob and lob['bids']['n'] > 0:
                        inventory_value = lob['bids']['best']  # Current market value
                    else:
                        inventory_value = trader.last_purchase_price  # Fallback to purchase price
                elif hasattr(trader, 'inventory') and trader.inventory > 0:
                    # For LLM trader that tracks inventory explicitly
                    if lob and lob['bids']['n'] > 0:
                        inventory_value = lob['bids']['best']  # Current market value
                    else:
                        inventory_value = trader.last_purchase_price if hasattr(trader, 'last_purchase_price') and trader.last_purchase_price is not None else 0
            
            net_worth += inventory_value
            net_worths[trader.ttype] = net_worth
    
    return net_worths


def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, dumpfile_flags, sess_vrbs):
    """
    One session in the market.
    :param sess_id: the character-string ID for this session, used in naming output files.
    :param starttime: the time the session starts.
    :param endtime: the time the sessiom ends.
    :param trader_spec: specification of the traders populating the market for this session.
    :param order_schedule: specification of the "customer orders" assigned to traders, i.e. the supply/demand schedule.
    :param dumpfile_flags: a dictionary of Boolean flags specifying which output files to be written for this session.
    :param sess_vrbs: verbosity: if True, output a running commentary on what is going on; if False, stay silent.
    :return: <nothing>.
    """

    def dump_strats_frame(frametime, stratfile, trdrs):
        """
        Write one frame of strategy snapshot
        :param frametime: the time that the frame snapshot is printed.
        :param stratfile:  the file to write to.
        :param trdrs: the population of traders.
        :return: <nothing>
        """

        line_str = 't=,%.0f, ' % frametime

        best_buyer_id = None
        best_buyer_prof = 0
        best_buyer_strat = None
        best_seller_id = None
        best_seller_prof = 0
        best_seller_strat = None

        # loop through traders to find the best
        for trdr in traders:
            trader = trdrs[trdr]

            # print('PRSH/PRDE/ZIPSH strategy recording, t=%s' % trader)
            if trader.ttype == 'PRSH' or trader.ttype == 'PRDE' or trader.ttype == 'ZIPSH':
                line_str += 'id=,%s, %s,' % (trader.tid, trader.ttype)

                if trader.ttype == 'ZIPSH':
                    # we know that ZIPSH sorts the set of strats into best-first
                    act_strat = trader.strats[0]['stratvec']
                    act_prof = trader.strats[0]['pps']
                else:
                    act_strat = trader.strats[trader.active_strat]['stratval']
                    act_prof = trader.strats[trader.active_strat]['pps']

                line_str += 'actvstrat=,%s ' % trader.strat_csv_str(act_strat)
                line_str += 'actvprof=,%f, ' % act_prof

                if trader.tid[:1] == 'B':
                    # this trader is a buyer
                    if best_buyer_id is None or act_prof > best_buyer_prof:
                        best_buyer_id = trader.tid
                        best_buyer_strat = act_strat
                        best_buyer_prof = act_prof
                elif trader.tid[:1] == 'S':
                    # this trader is a seller
                    if best_seller_id is None or act_prof > best_seller_prof:
                        best_seller_id = trader.tid
                        best_seller_strat = act_strat
                        best_seller_prof = act_prof
                else:
                    # wtf?
                    sys.exit('unknown trader id type in market_session')

        if best_buyer_id is not None:
            line_str += 'best_B_id=,%s, best_B_prof=,%f, best_B_strat=, ' % (best_buyer_id, best_buyer_prof)
            line_str += traders[best_buyer_id].strat_csv_str(best_buyer_strat)

        if best_seller_id is not None:
            line_str += 'best_S_id=,%s, best_S_prof=,%f, best_S_strat=, ' % (best_seller_id, best_seller_prof)
            line_str += traders[best_seller_id].strat_csv_str(best_seller_strat)

        line_str += '\n'

        if verbose:
            print('line_str: %s' % line_str)
        stratfile.write(line_str)
        stratfile.flush()
        os.fsync(stratfile)

    def blotter_dump(session_id, trdrs):
        """
        Write the blotter for each trader.
        :param session_id: this market session's ID string (used for the filename).
        :param trdrs: the population of traders.
        :return: <nothing>
        """
        bdump = open(session_id+'_blotters.csv', 'w')
        for trdr in trdrs:
            bdump.write('%s, %d\n' % (trdrs[trdr].tid, len(trdrs[trdr].blotter)))
            for b in trdrs[trdr].blotter:
                bdump.write('%s, %s, %.3f, %d, %s, %s, %d\n'
                            % (traders[trdr].tid, b['type'], b['time'], b['price'], b['party1'], b['party2'], b['qty']))
        bdump.close()

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    if dumpfile_flags['dump_strats']:
        strat_dump = open(sess_id + '_strats.csv', 'w')
    else:
        strat_dump = None

    if dumpfile_flags['dump_lobs']:
        lobframes = open(sess_id + '_LOB_frames.csv', 'w')
    else:
        lobframes = None

    if dumpfile_flags['dump_avgbals']:
        avg_bals = open(sess_id + '_avg_balance.csv', 'w')
    else:
        avg_bals = None
        
    if dumpfile_flags['dump_tape']:
        # NB writing transactions only -- not writing cancellations
        tape_dump = open(sess_id + '_tape.csv', 'w')
    else:
        tape_dump = None
    
    # Initialize proprietary trader net worth tracking
    prop_net_worth_file = open(sess_id + '_prop_net_worths.csv', 'w')
    prop_net_worth_writer = csv.writer(prop_net_worth_file)
    # prop_net_worth_writer.writerow(['Timestamp', 'PT1_NetWorth', 'PT2_NetWorth', 'LLM_NetWorth', 'BG_NetWorth', 'BGNO_NetWorth', 'PGCO_NetWorth', 'PGNO_NetWorth', 'GV1_NetWorth', 'GV2_NetWorth'])

    prop_net_worth_writer.writerow(PROP_TRADER_CSV_HEADERS)
        
    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, populate_verbose)

    # Set traders_dict for perfect belief graph traders
    for tid, trader in traders.items():
        if hasattr(trader, 'set_traders_dict'):
            trader.set_traders_dict(traders)

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 10.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'] + trader_stats['n_proptraders'])  # increased by 10x to reduce iterations

    session_duration = float(endtime - starttime)

    time = starttime

    pending_cust_orders = []

    if sess_vrbs:
        print('\n%s;  ' % sess_id)

    # frames_done is record of what frames we have printed data for thus far
    frames_done = set()

    # Progress tracking
    total_timesteps = int((endtime - starttime) / timestep)
    current_timestep = 0

    while time < endtime:

        time_left = (endtime - time) / session_duration
        
        current_timestep += 1
        if current_timestep % 100 == 0:  # Every 1k timesteps
            print(f"Progress: {current_timestep:,}/{total_timesteps:,} ({current_timestep/total_timesteps*100:.1f}%)")

        [pending_cust_orders, kills] = customer_orders(time, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    # NB if exchange.del_order() third argument = None then cancellations not written to tape file.
                    # exchange.del_order(time, traders[kill].lastquote, tape_dump, sess_vrbs)
                    exchange.del_order(time, traders[kill].lastquote, None, sess_vrbs)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]

        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lobframes, lob_verbose))
        if sess_vrbs:
            print('trader=%s order=%s' % (tid, order))

        if order is not None:
            # Only validate customer traders (buyers/sellers), not proprietary traders
            if tid[0] != 'P' and len(traders[tid].orders) > 0:
                if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                    sys.exit('Bad ask')
                if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                    sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order(time, order, tape_dump, process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(time, trade, order, bookkeep_verbose)
                traders[trade['party2']].bookkeep(time, trade, order, bookkeep_verbose)
                if dumpfile_flags['dump_avgbals']:
                    trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose))
                
                # Record proprietary trader net worths
                net_worths = calculate_prop_trader_net_worth(traders, lob)
                row = [int(time)] + [net_worths.get(ttype, 500) for ttype in PROP_TRADER_TYPES]
                prop_net_worth_writer.writerow(row)

            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lobframes, lob_verbose)
            any_record_frame = False
            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                record_frame = traders[t].respond(time, lob, trade, respond_verbose)
                if record_frame:
                    any_record_frame = True

            # log all the PRSH/PRDE/ZIPSH strategy info for this timestep?
            if any_record_frame and dumpfile_flags['dump_strats']:
                # print one more frame to strategy dumpfile
                dump_strats_frame(time, strat_dump, traders)
                # record that we've written this frame
                frames_done.add(int(time))

        time = time + timestep

    # session has ended

    # write trade_stats for this session (NB could use this to write end-of-session summary only)
    if dumpfile_flags['dump_avgbals']:
        trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose))
        avg_bals.close()

    if dumpfile_flags['dump_blotters']:
        # record the blotter for each trader
        blotter_dump(sess_id, traders)

    if dumpfile_flags['dump_strats']:
        strat_dump.close()

    if dumpfile_flags['dump_lobs']:
        lobframes.close()

    # Close proprietary trader net worth file
    prop_net_worth_file.close()

    # Print net worths of all proprietary traders
    print("\n" + "="*60)
    print("PROPRIETARY TRADER NET WORTHS")
    print("="*60)
    
    # Get final LOB for current market prices
    final_lob = exchange.publish_lob(time, lobframes, lob_verbose)
    
    prop_traders = []
    for tid, trader in traders.items():
        # if trader.ttype in ['PT1', 'PT2', 'LLM', 'BG', 'BGNO', 'PGCO', 'PGNO', 'GV1', 'GV2']:

        if trader.ttype in PROP_TRADER_TYPES:
            net_worth = trader.balance
            
            # Check if trader is holding inventory
            inventory_value = 0
            if hasattr(trader, 'job') and trader.job == 'Sell':
                # Trader is in sell mode, so they have inventory
                if hasattr(trader, 'last_purchase_price') and trader.last_purchase_price is not None:
                    # Use current market value (best bid) instead of purchase price
                    if final_lob and final_lob['bids']['n'] > 0:
                        inventory_value = final_lob['bids']['best']  # Current market value
                    else:
                        inventory_value = trader.last_purchase_price  # Fallback to purchase price
                elif hasattr(trader, 'inventory') and trader.inventory > 0:
                    # For LLM trader that tracks inventory explicitly
                    if final_lob and final_lob['bids']['n'] > 0:
                        inventory_value = final_lob['bids']['best']  # Current market value
                    else:
                        inventory_value = trader.last_purchase_price if hasattr(trader, 'last_purchase_price') and trader.last_purchase_price is not None else 0
            
            net_worth += inventory_value
            prop_traders.append((tid, trader.ttype, net_worth, trader.balance, inventory_value))
    
    # Sort by net worth (descending)
    prop_traders.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (tid, ttype, net_worth, cash, inventory) in enumerate(prop_traders, 1):
        if inventory > 0:
            print(f"{rank}. {tid} ({ttype}): ${net_worth} (${cash} cash + ${inventory} inventory)")
        else:
            print(f"{rank}. {tid} ({ttype}): ${net_worth} (${cash} cash)")
    
    print("="*60)


#############################
# # Below here is where we set up and run a whole series of experiments

# ============================================================================
# GLOBAL TRADER CONFIGURATION
# ============================================================================
# Edit these lists to easily control which traders are included in the simulation

# Available trader types with their default parameters and class mappings
AVAILABLE_TRADER_TYPES = {
    # Standard algorithmic traders (use $0 starting balance)
    'SHVR': {'class': 'TraderShaver', 'balance_type': 'standard', 'params': {}},
    'GVWY': {'class': 'TraderGiveaway', 'balance_type': 'standard', 'params': {}},
    'ZIC': {'class': 'TraderZIC', 'balance_type': 'standard', 'params': {}},
    'ZIP': {'class': 'TraderZIP', 'balance_type': 'standard', 'params': {}},
    'ZIPSH': {'class': 'TraderZIP', 'balance_type': 'standard', 'params': {}},
    'SNPR': {'class': 'TraderSniper', 'balance_type': 'standard', 'params': {}},
    'PRZI': {'class': 'TraderPRZI', 'balance_type': 'standard', 'params': {}},
    'PRSH': {'class': 'TraderPRZI', 'balance_type': 'standard', 'params': {}},
    'PRDE': {'class': 'TraderPRZI', 'balance_type': 'standard', 'params': {}},

    # Proprietary traders - Traditional (use $500 starting balance)
    'PT1': {'class': 'TraderPT1', 'balance_type': 'prop', 'params': {'bid_percent': 0.95, 'ask_delta': 2, 'n_past_trades': 5}},
    'PT2': {'class': 'TraderPT2', 'balance_type': 'prop', 'params': {'bid_percent': 0.99, 'ask_delta': 2, 'n_past_trades': 5}},

    # Proprietary traders - LLM-Based (all 25 variants from AgentFactory)
    'LLM': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'BG_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'BG_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'BG_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'BG_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'PG_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'PG_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'PG_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'PG_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV1_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV1_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV1_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV1_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV2_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV2_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV2_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV2_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV3_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV3_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV3_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'GV3_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'HM_JSON_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'HM_NL_COT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'HM_JSON_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
    'HM_NL_NOCOT': {'class': 'AgentFactory', 'balance_type': 'prop', 'params': {}},
}

# CONFIGURATION: Edit these to control which traders are included
ACTIVE_BUYERS = [('SHVR', 5), ('GVWY', 5), ('ZIC', 2), ('ZIP', 11)]
ACTIVE_SELLERS = [('SHVR', 5), ('GVWY', 5), ('ZIC', 2), ('ZIP', 11)]  # Usually same as buyers
ACTIVE_PROPTRADERS = [
    ('LLM', 1),
    ('BG_JSON_COT', 1),
    ('BG_JSON_NOCOT', 1),
    ('BG_NL_COT', 1),
    ('BG_NL_NOCOT', 1),
    ('GV1_JSON_COT', 1),
    ('GV1_JSON_NOCOT', 1),
    ('GV1_NL_COT', 1),
    ('GV1_NL_NOCOT', 1),
    ('GV2_JSON_COT', 1),
    ('GV2_JSON_NOCOT', 1),
    ('GV2_NL_COT', 1),
    ('GV2_NL_NOCOT', 1),
    ('GV3_JSON_COT', 1),
    ('GV3_JSON_NOCOT', 1),
    ('GV3_NL_COT', 1),
    ('GV3_NL_NOCOT', 1),
    ('HM_JSON_COT', 1),
    ('HM_JSON_NOCOT', 1),
    ('HM_NL_COT', 1),
    ('HM_NL_NOCOT', 1),
]  # 21 LLM agents (1 baseline + 5 belief graph types * 4 formats)

# Automatically generate lists of proprietary trader types for filtering
PROP_TRADER_TYPES = [ttype for ttype, count in ACTIVE_PROPTRADERS]
PROP_TRADER_CSV_HEADERS = ['Timestamp'] + [f'{ttype}_NetWorth' for ttype in PROP_TRADER_TYPES]

# ============================================================================

def get_trader_parameters(trader_type):
    """Get default parameters for a trader type"""
    trader_info = AVAILABLE_TRADER_TYPES.get(trader_type, {})
    return trader_info.get('params', {})

def validate_trader_configuration():
    """Validate that all configured trader types are available"""
    all_configured = []
    
    # Check buyers
    for ttype, count in ACTIVE_BUYERS:
        all_configured.append(ttype)
        if ttype not in AVAILABLE_TRADER_TYPES:
            raise ValueError(f"Unknown buyer trader type: {ttype}")
    
    # Check sellers  
    for ttype, count in ACTIVE_SELLERS:
        all_configured.append(ttype)
        if ttype not in AVAILABLE_TRADER_TYPES:
            raise ValueError(f"Unknown seller trader type: {ttype}")
            
    # Check prop traders
    for ttype, count in ACTIVE_PROPTRADERS:
        all_configured.append(ttype)
        if ttype not in AVAILABLE_TRADER_TYPES:
            raise ValueError(f"Unknown proprietary trader type: {ttype}")
    
    print(f"✓ Configuration validated. Active trader types: {sorted(set(all_configured))}")
    return True

if __name__ == "__main__":

    # Validate configuration before starting
    validate_trader_configuration()
    
    print(f"Active proprietary traders: {PROP_TRADER_TYPES}")
    print(f"CSV headers: {PROP_TRADER_CSV_HEADERS}")

    price_offset_filename = 'offset_BTC_USD_20250325.csv'

    # if called from the command line with one argument, the first argument is the price offset filename
    if len(sys.argv) > 1:
        price_offset_filename = sys.argv[1]

    # set up common parameters for all market sessions
    n_days = 1/1440
    hours_in_a_day = 24     # how many hours the exchange operates for in a working day (e.g. NYSE = 7.5)
    start_time = 0.0
    end_time = 60.0 * 60.0 * hours_in_a_day * n_days
    duration = end_time - start_time


    def schedule_offsetfn_read_file(filename, col_t, col_p, scale_factor=75):
        """
        Read in a CSV data-file for the supply/demand schedule time-varying price-offset value
        :param filename: the CSV file to read
        :param col_t: column in the CSV that has the time data
        :param col_p: column in the CSV that has the price data
        :param scale_factor: multiplier on prices
        :return: on offset value event-list: one item for each change in offset value
                -- each item is percentage time elapsed, followed by the new offset value at that time
        """
        
        vrbs = True
        
        # does two passes through the file
        # assumes data file is all for one date, sorted in time order, in correct format, etc. etc.
        rwd_csv = csv.reader(open(filename, 'r'))
        
        # first pass: get time & price events, find out how long session is, get min & max price
        minprice = None
        maxprice = None
        firsttimeobj = None
        timesincestart = 0
        priceevents = []
        
        first_row_is_header = True
        this_is_first_row = True
        this_is_first_data_row = True
        first_date = None
        
        for line in rwd_csv:
            
            if vrbs:
                print(line)
            
            if this_is_first_row and first_row_is_header:
                this_is_first_row = False
                this_is_first_data_row = True
                continue
                
            row_date = line[col_t][:10]
            
            if this_is_first_data_row:
                first_date = row_date
                this_is_first_data_row = False
                
            if row_date != first_date:
                continue
                
            time = line[col_t][11:19]
            if firsttimeobj is None:
                firsttimeobj = datetime.strptime(time, '%H:%M:%S')
                
            timeobj = datetime.strptime(time, '%H:%M:%S')
            
            price_str = line[col_p]
            # delete any commas so 1,000,000 becomes 1000000
            price_str_no_commas = price_str.replace(',', '')
            price = float(price_str_no_commas)
            
            if minprice is None or price < minprice:
                minprice = price
            if maxprice is None or price > maxprice:
                maxprice = price
            timesincestart = (timeobj - firsttimeobj).total_seconds()
            priceevents.append([timesincestart, price])
            
            if vrbs:
                print(row_date, time, timesincestart, price)
            
        # second pass: normalise times to fractions of entire time-series duration
        #              & normalise price range
        pricerange = maxprice - minprice
        endtime = float(timesincestart)
        offsetfn_eventlist = []
        for event in priceevents:
            # normalise price
            normld_price = (event[1] - minprice) / pricerange
            # clip
            normld_price = min(normld_price, 1.0)
            normld_price = max(0.0, normld_price)
            # scale & convert to integer cents
            price = int(round(normld_price * scale_factor))
            normld_event = [event[0] / endtime, price]
            if vrbs:
                print(normld_event)
            offsetfn_eventlist.append(normld_event)
        
        return offsetfn_eventlist


    def schedule_offsetfn_from_eventlist(time, params):
        """
        Returns a price offset-value for the current time, by reading from an offset event-list.
        :param time: the current time
        :param params: a list of parameter values...
            params[1] is the final time (the end-time) of the current session.
            params[2] is the offset event-list: one item for each change in offset value
                        -- each item is percentage time elapsed, followed by the new offset value at that time
        :return: integer price offset value
        """

        final_time = float(params[0])
        offset_events = params[1]
        # this is quite inefficient: on every call it walks the event-list
        percent_elapsed = time/final_time
        offset = None
        for event in offset_events:
            offset = event[1]
            if percent_elapsed < event[0]:
                break
        return offset


    def schedule_offsetfn_increasing_sinusoid(t, params):
        """
        Returns sinusoidal time-dependent price-offset, steadily increasing in frequency & amplitude
        :param t: time
        :param params: set of parameters for the offsetfn: this is empty-set for this offsetfn but nonempty in others
        :return: the time-dependent price offset at time t
        """
        if params is None:  # this test of params is here only to prevent PyCharm from warning about unused parameters
            pass
        scale = -7500
        multiplier = 7500000    # determines rate of increase of frequency and amplitude
        offset = ((scale * t) / multiplier) * (1 + math.sin((t*t)/(multiplier * math.pi)))
        return int(round(offset, 0))

    # Here is an example of how to use the offset function
    #
    # range1 = (10, 190, (schedule_offsetfn, args)) # args is the list of arguments to the function
    # range2 = (200, 300, (schedule_offsetfn, args))

    # Here is an example of how to switch from range1 to range2 and then back to range1,
    # introducing two "market shocks"
    # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.
    #
    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #                   ]

    offsetfn_events = None
    if price_offset_filename is not None:
        offsetfn_events = schedule_offsetfn_read_file(price_offset_filename, 0, 1)

    # supply schedule (defines the supply curve)
    range1 = (75, 110, (schedule_offsetfn_from_eventlist, [[end_time, offsetfn_events]]))
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'random'}]

    # demand schedule (defines the demand curve)
    range2 = (125, 90, (schedule_offsetfn_from_eventlist, [[end_time, offsetfn_events]]))
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'random'}]

    # new customer orders arrive at each trader approx once every order_interval seconds
    order_interval = 10

    # order schedule wraps up the supply/demand schedules and details of how customer orders/assignments are issued
    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': order_interval, 'timemode': 'drip-poisson'}

    # now run a sequence of trials, one session per trial

    # if verbose = True, print a running commentary describing what's going on.
    verbose = False

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 1

    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 5

    trial = 1

    while trial < (n_trials+1):

        # create unique i.d. string for this trial
        trial_id = 'bse_d%03d_i%02d_%04d' % (n_days, order_interval, trial)

        # Use the unified configuration system
        buyers_spec = ACTIVE_BUYERS
        sellers_spec = ACTIVE_SELLERS  
        proptraders_spec = ACTIVE_PROPTRADERS

        # trader_spec wraps up the specifications for the buyers, sellers, and proptraders
        traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec, 'proptraders': proptraders_spec}

        if trial > n_trials_recorded:
            # switch off recording of detailed data-files
            dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                          'dump_avgbals': False, 'dump_tape': False}
        else:
            # we're still recording all the required data-files
            dump_flags = {'dump_blotters': True, 'dump_lobs': False, 'dump_strats': True,
                          'dump_avgbals': True, 'dump_tape': True}

        # simulate the market session
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

        trial = trial + 1

    # The code in comments below here is for illustration, in case you want to do an exhaustive sweep of all possible
    # combinations of some set of trading strategies: if its of no interest, it can be deleted.
    #
    # run a sequence of trials that exhaustively varies the ratio of four trader types
    # NB this has weakness of symmetric proportions on buyers/sellers -- combinatorics of varying that are quite nasty
    #
    # n_trader_types = 4
    # equal_ratio_n = 4
    # n_trials_per_ratio = 50
    #
    # n_traders = n_trader_types * equal_ratio_n
    #
    # fname = 'balances_%03d.csv' % equal_ratio_n
    #
    # tdump = open(fname, 'w')
    #
    # min_n = 1
    #
    # trialnumber = 1
    # trdr_1_n = min_n
    # while trdr_1_n <= n_traders:
    #     trdr_2_n = min_n
    #     while trdr_2_n <= n_traders - trdr_1_n:
    #         trdr_3_n = min_n
    #         while trdr_3_n <= n_traders - (trdr_1_n + trdr_2_n):
    #             trdr_4_n = n_traders - (trdr_1_n + trdr_2_n + trdr_3_n)
    #             if trdr_4_n >= min_n:
    #                 buyers_spec = [('GVWY', trdr_1_n), ('SHVR', trdr_2_n),
    #                                ('ZIC', trdr_3_n), ('ZIP', trdr_4_n)]
    #                 sellers_spec = buyers_spec
    #                 traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}
    #                 # print buyers_spec
    #                 trial = 1
    #                 while trial <= n_trials_per_ratio:
    #                     trial_id = 'trial%07d' % trialnumber
    #                     market_session(trial_id, start_time, end_time, traders_spec,
    #                                    order_sched, tdump, False, True)
    #                     tdump.flush()
    #                     trial = trial + 1
    #                     trialnumber = trialnumber + 1
    #             trdr_3_n += 1
    #         trdr_2_n += 1
    #     trdr_1_n += 1
    # tdump.close()
    #
    # print(trialnumber)
