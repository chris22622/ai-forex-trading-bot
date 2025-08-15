#!/usr/bin/env python3
"""
Test script for the enhanced multi-symbol trading bot
Tests all new features including symbol scanning and full trade cycles
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DerivTradingBot, PROFIT_SYMBOL_UNIVERSE, TradeAction
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class MultiSymbolBotTester:
    """Test suite for the enhanced multi-symbol trading bot"""
    
    def __init__(self):
        self.bot = None
        self.test_results = {
            'symbol_initialization': False,
            'market_scanning': False,
            'opportunity_detection': False,
            'trade_execution': False,
            'position_management': False,
            'full_trade_cycle': False,
            'profit_calculation': False
        }
    
    async def run_all_tests(self) -> bool:
        """Run comprehensive test suite"""
        logger.info("🧪 Starting Multi-Symbol Trading Bot Test Suite")
        logger.info("=" * 60)
        
        try:
            # Initialize bot
            self.bot = DerivTradingBot()
            logger.info("✅ Bot initialized successfully")
            
            # Test 1: Symbol Universe Initialization
            await self.test_symbol_initialization()
            
            # Test 2: Market Data Scanning
            await self.test_market_scanning()
            
            # Test 3: Opportunity Detection
            await self.test_opportunity_detection()
            
            # Test 4: Trade Execution
            await self.test_trade_execution()
            
            # Test 5: Position Management
            await self.test_position_management()
            
            # Test 6: Full Trade Cycle
            await self.test_full_trade_cycle()
            
            # Test 7: Profit Calculations
            await self.test_profit_calculations()
            
            # Display results
            self.display_test_results()
            
            return all(self.test_results.values())
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False
    
    async def test_symbol_initialization(self) -> None:
        """Test symbol universe initialization"""
        logger.info("🔍 Testing symbol universe initialization...")
        
        try:
            # Test symbol universe structure
            assert isinstance(PROFIT_SYMBOL_UNIVERSE, dict), "Symbol universe should be a dictionary"
            
            total_symbols = sum(len(symbols) for symbols in PROFIT_SYMBOL_UNIVERSE.values())
            logger.info(f"📊 Total symbols in universe: {total_symbols}")
            
            # Test categories
            expected_categories = ['indices', 'forex', 'commodities', 'crypto', 'stock_indices']
            for category in expected_categories:
                assert category in PROFIT_SYMBOL_UNIVERSE, f"Missing category: {category}"
                logger.info(f"✅ Category '{category}': {len(PROFIT_SYMBOL_UNIVERSE[category])} symbols")
            
            # Initialize symbol data structures
            await self.bot.initialize_symbol_universe()
            
            assert hasattr(self.bot, 'active_symbols'), "Bot should have active_symbols"
            assert hasattr(self.bot, 'symbol_data'), "Bot should have symbol_data"
            assert hasattr(self.bot, 'symbol_performance'), "Bot should have symbol_performance"
            
            logger.info(f"✅ Initialized {len(self.bot.active_symbols)} active symbols")
            
            self.test_results['symbol_initialization'] = True
            
        except Exception as e:
            logger.error(f"❌ Symbol initialization test failed: {e}")
    
    async def test_market_scanning(self) -> None:
        """Test continuous market scanning functionality"""
        logger.info("🔍 Testing market scanning functionality...")
        
        try:
            # Ensure we have some active symbols
            if not self.bot.active_symbols:
                self.bot.active_symbols = ['R_50', 'R_75', 'EURUSD', 'XAUUSD']
            
            # Initialize symbol data for testing
            for symbol in self.bot.active_symbols[:3]:  # Test with first 3 symbols
                await self.bot.update_symbol_data(symbol)
                
                # Verify symbol data structure
                assert symbol in self.bot.symbol_data, f"Symbol {symbol} should be in symbol_data"
                
                symbol_data = self.bot.symbol_data[symbol]
                expected_fields = ['last_price', 'price_history', 'volatility', 'trend_strength', 'profit_score']
                
                for field in expected_fields:
                    assert field in symbol_data, f"Symbol data should contain {field}"
                
                logger.info(f"✅ {symbol}: price={symbol_data['last_price']:.4f}, volatility={symbol_data['volatility']:.4f}")
            
            logger.info("✅ Market scanning functionality working")
            self.test_results['market_scanning'] = True
            
        except Exception as e:
            logger.error(f"❌ Market scanning test failed: {e}")
    
    async def test_opportunity_detection(self) -> None:
        """Test profit opportunity detection"""
        logger.info("🔍 Testing opportunity detection...")
        
        try:
            opportunities_found = 0
            
            for symbol in self.bot.active_symbols[:5]:  # Test with first 5 symbols
                # Simulate some price history for analysis
                if symbol in self.bot.symbol_data:
                    symbol_data = self.bot.symbol_data[symbol]
                    
                    # Add some simulated price history
                    base_price = symbol_data['last_price']
                    for i in range(20):
                        # Create trending price movement
                        price = base_price * (1 + (i * 0.001))  # 0.1% trend
                        symbol_data['price_history'].append(price)
                    
                    # Update calculated fields
                    symbol_data['volatility'] = 0.15  # Good volatility
                    symbol_data['trend_strength'] = 0.7  # Strong uptrend
                    symbol_data['profit_score'] = 45  # Good profit score
                    
                    # Test opportunity detection
                    opportunity = await self.bot.analyze_profit_opportunity(symbol)
                    
                    if opportunity:
                        opportunities_found += 1
                        logger.info(f"🎯 Opportunity found: {symbol} - {opportunity['action'].value} (confidence: {opportunity['confidence']:.0%})")
                        
                        # Verify opportunity structure
                        expected_fields = ['symbol', 'action', 'confidence', 'profit_score', 'volatility']
                        for field in expected_fields:
                            assert field in opportunity, f"Opportunity should contain {field}"
            
            logger.info(f"✅ Found {opportunities_found} trading opportunities")
            assert opportunities_found > 0, "Should find at least one opportunity"
            
            self.test_results['opportunity_detection'] = True
            
        except Exception as e:
            logger.error(f"❌ Opportunity detection test failed: {e}")
    
    async def test_trade_execution(self) -> None:
        """Test trade execution functionality"""
        logger.info("🔍 Testing trade execution...")
        
        try:
            # Test with a simulated opportunity
            test_symbol = 'R_50'
            test_opportunity = {
                'symbol': test_symbol,
                'action': TradeAction.BUY,
                'confidence': 0.85,
                'profit_score': 50,
                'volatility': 0.12,
                'priority': 'HIGH'
            }
            
            # Execute the opportunity
            success = await self.bot.execute_trade_opportunity(test_opportunity)
            
            assert success, "Trade execution should succeed"
            
            # Verify position was created
            assert test_symbol in self.bot.open_positions, f"Position should be created for {test_symbol}"
            
            position = self.bot.open_positions[test_symbol]
            expected_fields = ['contract_id', 'action', 'symbol', 'amount', 'entry_price', 'confidence']
            
            for field in expected_fields:
                assert field in position, f"Position should contain {field}"
            
            logger.info(f"✅ Trade executed: {test_symbol} {position['action']} ${position['amount']:.2f}")
            
            self.test_results['trade_execution'] = True
            
        except Exception as e:
            logger.error(f"❌ Trade execution test failed: {e}")
    
    async def test_position_management(self) -> None:
        """Test position monitoring and management"""
        logger.info("🔍 Testing position management...")
        
        try:
            # Should have at least one position from previous test
            assert len(self.bot.open_positions) > 0, "Should have open positions"
            
            # Test position status update
            for symbol, position in self.bot.open_positions.items():
                await self.bot.update_position_status(symbol, position)
                
                # Verify position was updated
                assert 'unrealized_pnl' in position, "Position should have unrealized P&L"
                assert 'current_price' in position, "Position should have current price"
                
                logger.info(f"✅ Position {symbol}: P&L = ${position.get('unrealized_pnl', 0):.2f}")
                
                # Test exit condition checking
                should_close = await self.bot.check_position_exit_conditions(symbol, position)
                logger.info(f"✅ Exit check for {symbol}: {should_close}")
            
            self.test_results['position_management'] = True
            
        except Exception as e:
            logger.error(f"❌ Position management test failed: {e}")
    
    async def test_full_trade_cycle(self) -> None:
        """Test complete trade cycle (open -> monitor -> close)"""
        logger.info("🔍 Testing full trade cycle...")
        
        try:
            test_symbol = 'R_75'
            
            # 1. Open a position
            test_opportunity = {
                'symbol': test_symbol,
                'action': TradeAction.SELL,
                'confidence': 0.75,
                'profit_score': 40,
                'volatility': 0.08,
                'priority': 'NORMAL'
            }
            
            open_success = await self.bot.execute_trade_opportunity(test_opportunity)
            assert open_success, "Should successfully open position"
            
            # 2. Monitor position
            await self.bot.update_position_status(test_symbol, self.bot.open_positions[test_symbol])
            
            # 3. Close position
            close_action = TradeAction.CLOSE_SELL
            close_success = await self.bot.close_position(test_symbol, close_action)
            assert close_success, "Should successfully close position"
            
            # 4. Verify position was removed
            assert test_symbol not in self.bot.open_positions, "Position should be closed"
            
            # 5. Verify trade cycle was recorded
            if test_symbol in self.bot.trade_cycles:
                assert len(self.bot.trade_cycles[test_symbol]) > 0, "Trade cycle should be recorded"
                cycle = self.bot.trade_cycles[test_symbol][-1]
                assert cycle['status'] == 'CLOSED', "Trade cycle should be marked as closed"
                logger.info(f"✅ Trade cycle completed: {cycle['result']} ${cycle.get('profit', 0):.2f}")
            
            self.test_results['full_trade_cycle'] = True
            
        except Exception as e:
            logger.error(f"❌ Full trade cycle test failed: {e}")
    
    async def test_profit_calculations(self) -> None:
        """Test profit calculations and performance tracking"""
        logger.info("🔍 Testing profit calculations...")
        
        try:
            test_symbol = 'EURUSD'
            
            # Test symbol performance tracking
            self.bot.update_symbol_performance(test_symbol, True, 15.50)  # Win
            self.bot.update_symbol_performance(test_symbol, False, -8.25)  # Loss
            self.bot.update_symbol_performance(test_symbol, True, 22.75)  # Win
            
            perf = self.bot.symbol_performance[test_symbol]
            
            assert perf['total_trades'] == 3, "Should have 3 trades"
            assert perf['wins'] == 2, "Should have 2 wins"
            assert perf['losses'] == 1, "Should have 1 loss"
            assert perf['win_rate'] == 2/3, "Win rate should be 66.67%"
            assert perf['total_profit'] == 30.0, "Total profit should be $30.00"
            
            logger.info(f"✅ Performance tracking: {perf['wins']}/{perf['total_trades']} wins, ${perf['total_profit']:.2f} profit")
            
            # Test dynamic position sizing
            base_amount = 10.0
            confidence = 0.8
            win_rate = 0.7
            
            dynamic_size = self.bot.calculate_dynamic_position_size(base_amount, confidence, win_rate)
            assert dynamic_size > base_amount, "Dynamic sizing should increase position for high confidence"
            
            logger.info(f"✅ Dynamic sizing: ${base_amount:.2f} -> ${dynamic_size:.2f} (confidence: {confidence:.0%})")
            
            self.test_results['profit_calculation'] = True
            
        except Exception as e:
            logger.error(f"❌ Profit calculation test failed: {e}")
    
    def display_test_results(self) -> None:
        """Display comprehensive test results"""
        logger.info("=" * 60)
        logger.info("🧪 MULTI-SYMBOL BOT TEST RESULTS")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {test_name.replace('_', ' ').title()}")
        
        logger.info("-" * 60)
        logger.info(f"📊 SUMMARY: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Multi-symbol bot is ready for maximum profit!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review and fix issues before trading.")
        
        logger.info("=" * 60)

async def main():
    """Main test execution"""
    print("🚀 MAXIMUM PROFIT MULTI-SYMBOL BOT TEST SUITE")
    print("Testing enhanced multi-symbol scanning and full trade cycles...")
    print()
    
    tester = MultiSymbolBotTester()
    success = await tester.run_all_tests()
    
    print()
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🚀 Your enhanced multi-symbol bot is ready for MAXIMUM PROFITS!")
        print("💰 The bot will now scan ALL available symbols and execute full trade cycles")
        print("📊 Features tested:")
        print("   - Multi-symbol universe initialization")
        print("   - Real-time market scanning")
        print("   - Profit opportunity detection")
        print("   - Advanced trade execution")
        print("   - Position management & monitoring")
        print("   - Complete trade cycles (buy/sell)")
        print("   - Dynamic profit calculations")
        print()
        print("🎯 Ready to run: python main.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please review the test output and fix any issues before running the bot.")
    
    return success

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
