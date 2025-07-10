import re
import unittest
from typing import Dict, Any
from opto.optimizers.optoprime_v2 import extract_xml_like_data

"""
1. Nested Tag Handling: The parser now uses a stack-based approach to extract only top-level tags, ignoring nested ones:
- <value> containing nested <variable> tags → only extracts the top-level value
- <name> containing nested <name> tags → only extracts the top-level name text
- Complex multi-level nesting → correctly handles all levels

2. Edge Case Handling:
- Empty tags: Allows variables with empty values if <value> tag is present
- Missing tags: Only adds variables if both <name> and <value> tags are present
- Malformed XML: Handles unclosed <reasoning> tags gracefully
- Whitespace: Proper handling of leading/trailing whitespace
- Special characters: Handles < > & " ' characters correctly
- Duplicate variable names: Later variables override earlier ones

3. Comprehensive Test Coverage (13 tests):
- Basic parsing functionality
- Nested variable/name/value tags
- Multiple nested levels
- Empty tags
- Missing tags
- Malformed XML
- Special characters
- Whitespace handling
- Duplicate variable names
- No reasoning/variable tags scenarios
"""

class TestXMLParsing(unittest.TestCase):
    
    def test_basic_parsing(self):
        """Test basic parsing functionality"""
        text = """
        <reasoning>
        This is my reasoning for the changes.
        </reasoning>
        
        <variable>
        <name>var1</name>
        <value>value1</value>
        </variable>
        
        <variable>
        <name>var2</name>
        <value>value2</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'This is my reasoning for the changes.',
            'variables': {
                'var1': 'value1',
                'var2': 'value2'
            }
        }
        self.assertEqual(result, expected)
    
    def test_nested_variable_tags(self):
        """Test that only top-level variable tags are extracted"""
        text = """
        <reasoning>Reasoning here</reasoning>
        
        <variable>
        <name>outer_var</name>
        <value>
        <variable>
        <name>inner_var</name>
        <value>inner_value</value>
        </variable>
        outer_value
        </value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning here',
            'variables': {
                'outer_var': '<variable>\n        <name>inner_var</name>\n        <value>inner_value</value>\n        </variable>\n        outer_value'
            }
        }
        self.assertEqual(result, expected)
    
    def test_nested_name_tags(self):
        """Test that only top-level name tags are extracted"""
        text = """
        <reasoning>Reasoning here</reasoning>
        
        <variable>
        <name>
        <name>inner_name</name>
        outer_name
        </name>
        <value>some_value</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning here',
            'variables': {
                'outer_name': 'some_value'
            }
        }
        self.assertEqual(result, expected)
    
    def test_nested_value_tags(self):
        """Test that only top-level value tags are extracted"""
        text = """
        <reasoning>Reasoning here</reasoning>
        
        <variable>
        <name>var_name</name>
        <value>
        <value>inner_value</value>
        outer_value
        </value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning here',
            'variables': {
                'var_name': '<value>inner_value</value>\n        outer_value'
            }
        }
        self.assertEqual(result, expected)
    
    def test_multiple_nested_levels(self):
        """Test complex nested structure"""
        text = """
        <reasoning>Complex reasoning</reasoning>
        
        <variable>
        <name>level1_name</name>
        <value>
        <variable>
        <name>level2_name</name>
        <value>
        <variable>
        <name>level3_name</name>
        <value>level3_value</value>
        </variable>
        level2_value
        </value>
        </variable>
        level1_value
        </value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Complex reasoning',
            'variables': {
                'level1_name': '<variable>\n        <name>level2_name</name>\n        <value>\n        <variable>\n        <name>level3_name</name>\n        <value>level3_value</value>\n        </variable>\n        level2_value\n        </value>\n        </variable>\n        level1_value'
            }
        }
        self.assertEqual(result, expected)
    
    def test_empty_tags(self):
        """Test handling of empty tags"""
        text = """
        <reasoning></reasoning>
        
        <variable>
        <name></name>
        <value>some_value</value>
        </variable>
        
        <variable>
        <name>valid_name</name>
        <value></value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': '',
            'variables': {
                'valid_name': ''
            }
        }
        self.assertEqual(result, expected)
    
    def test_missing_tags(self):
        """Test handling of missing tags"""
        text = """
        <reasoning>Some reasoning</reasoning>
        
        <variable>
        <name>var1</name>
        </variable>
        
        <variable>
        <value>value2</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Some reasoning',
            'variables': {}
        }
        self.assertEqual(result, expected)
    
    def test_malformed_xml(self):
        """Test handling of malformed XML"""
        text = """
        <reasoning>Reasoning
        <variable>
        <name>var1</name>
        <value>value1</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning\n        <variable>\n        <name>var1</name>\n        <value>value1</value>\n        </variable>\n        ',
            'variables': {}
        }
        self.assertEqual(result, expected)
    
    def test_no_reasoning_tag(self):
        """Test when reasoning tag is missing"""
        text = """
        <variable>
        <name>var1</name>
        <value>value1</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': '',
            'variables': {
                'var1': 'value1'
            }
        }
        self.assertEqual(result, expected)
    
    def test_no_variable_tags(self):
        """Test when no variable tags are present"""
        text = """
        <reasoning>Just reasoning, no variables</reasoning>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Just reasoning, no variables',
            'variables': {}
        }
        self.assertEqual(result, expected)
    
    def test_whitespace_handling(self):
        """Test proper whitespace handling"""
        text = """
        <reasoning>
            Reasoning with
            multiple lines
        </reasoning>
        
        <variable>
            <name>  var_name  </name>
            <value>
                value with
                multiple lines
            </value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning with\n            multiple lines',
            'variables': {
                'var_name': 'value with\n                multiple lines'
            }
        }
        self.assertEqual(result, expected)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        text = """
        <reasoning>Reasoning with < > & " ' characters</reasoning>
        
        <variable>
        <name>var_with_special_chars</name>
        <value>Value with < > & " ' characters</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning with < > & " \' characters',
            'variables': {
                'var_with_special_chars': 'Value with < > & " \' characters'
            }
        }
        self.assertEqual(result, expected)
    
    def test_duplicate_variable_names(self):
        """Test that later variables override earlier ones with same name"""
        text = """
        <reasoning>Reasoning</reasoning>
        
        <variable>
        <name>duplicate_var</name>
        <value>first_value</value>
        </variable>
        
        <variable>
        <name>duplicate_var</name>
        <value>second_value</value>
        </variable>
        """
        
        result = extract_xml_like_data(text)
        expected = {
            'reasoning': 'Reasoning',
            'variables': {
                'duplicate_var': 'second_value'
            }
        }
        self.assertEqual(result, expected)

    def test_xml_with_random_text(self):
        """Test that parser extracts XML content while ignoring random text"""
        text = """
        This is some random texts with random symbols `~!@#$%^&*()-=[]\;',./_+{}|:"<>?. 
        
        <reasoning>
        Some reasoning. 
        </reasoning>
        
        Some other random texts with random symbols `~!@#$%^&*()-=[]\;',./_+{}|:"<>?. 
        
        <variable>
        <name>var1</name>
        <value>value1</value>
        </variable>
          
        Yet another random texts with random symbols `~!@#$%^&*()-=[]\;',./_+{}|:"<>?. 
        
        <variable>
        <name>var2</name>
        <value>value2</value>
        </variable>
        """
        
        result = extract_xml_like_data(text, name_tag="name", value_tag="value")
        expected = {
            'reasoning': 'Some reasoning.',
            'variables': {
                'var1': 'value1',
                'var2': 'value2'
            }
        }
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()