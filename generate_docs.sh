#!/bin/bash

echo "🚀 Generating comprehensive Hagike documentation..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create documentation directory structure
echo -e "${BLUE}📁 Setting up documentation structure...${NC}"
mkdir -p docs/manual
mkdir -p docs/reference
mkdir -p docs/examples

# Generate HTML documentation using pdoc
echo -e "${BLUE}🔧 Generating HTML API documentation...${NC}"
find docs/hagike -type f -name "*.html" -delete 2>/dev/null || true
pdoc --output-dir docs hagike --html --force --config latex_math=True

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ HTML documentation generated successfully${NC}"
else
    echo -e "${RED}❌ Error generating HTML documentation${NC}"
fi

# Copy our comprehensive markdown documentation
echo -e "${BLUE}📚 Organizing comprehensive documentation...${NC}"

# Main documentation files
cp API_DOCUMENTATION.md docs/manual/
cp QUICK_START_GUIDE.md docs/manual/
cp FUNCTION_REFERENCE.md docs/reference/
cp DOCUMENTATION_INDEX.md docs/

# Create a main index file for the docs directory
cat > docs/index.md << 'EOF'
# Hagike Toolkit Documentation

Welcome to the Hagike toolkit documentation! Choose your starting point:

## 🚀 Quick Access

- **[Documentation Index](DOCUMENTATION_INDEX.md)** - Navigate all documentation
- **[Quick Start Guide](manual/QUICK_START_GUIDE.md)** - Get started in 5 minutes
- **[Complete API Documentation](manual/API_DOCUMENTATION.md)** - Comprehensive reference
- **[Function Reference](reference/FUNCTION_REFERENCE.md)** - Quick lookup table

## 📖 Documentation Types

### For New Users
- Start with the [Quick Start Guide](manual/QUICK_START_GUIDE.md)
- Then explore the [API Documentation](manual/API_DOCUMENTATION.md)

### For Developers
- Use the [Function Reference](reference/FUNCTION_REFERENCE.md) for quick lookups
- Check the [HTML API docs](hagike/index.html) for detailed docstrings

### For Contributors
- Read the [API Documentation](manual/API_DOCUMENTATION.md) for design principles
- Follow the patterns in the [Function Reference](reference/FUNCTION_REFERENCE.md)

## 🔗 External Links

- [GitHub Repository](https://github.com/hagikehappy/hagike)
- [PyPI Package](https://pypi.org/project/hagike/)

---

*Generated automatically - see individual files for detailed information*
EOF

# Create a comprehensive README for the docs directory
cat > docs/README.md << 'EOF'
# Hagike Documentation

This directory contains comprehensive documentation for the Hagike toolkit.

## Structure

```
docs/
├── index.md                    # Main documentation index
├── DOCUMENTATION_INDEX.md      # Navigation guide
├── manual/                     # User guides and tutorials
│   ├── API_DOCUMENTATION.md    # Complete API reference
│   └── QUICK_START_GUIDE.md    # Quick start tutorial
├── reference/                  # Quick reference materials
│   └── FUNCTION_REFERENCE.md   # Function lookup table
├── hagike/                     # Auto-generated HTML docs
│   └── index.html             # HTML API documentation
└── README.md                  # This file
```

## Documentation Types

### Manual Documentation (Markdown)
- **Comprehensive**: Complete coverage of all APIs and features
- **Examples**: Practical code examples and usage patterns
- **Searchable**: Easy to search and navigate
- **Version Controlled**: Tracked in git with the code

### HTML Documentation (Auto-generated)
- **Detailed**: Generated from source code docstrings
- **Up-to-date**: Automatically reflects latest code changes
- **Interactive**: Browsable with cross-references
- **Technical**: Focused on API details

## Building Documentation

### HTML Documentation
```bash
# Generate HTML docs from source code
./doc.sh
```

### Complete Documentation Suite
```bash
# Generate all documentation (HTML + comprehensive guides)
./generate_docs.sh
```

## Maintenance

- **Manual docs**: Update when adding new features or changing APIs
- **HTML docs**: Regenerated automatically from docstrings
- **Examples**: Test examples to ensure they work with current code
- **Links**: Check internal links when restructuring

## Contributing

When adding new features:
1. Update source code docstrings
2. Add examples to appropriate manual pages
3. Update function reference if needed
4. Test documentation generation
5. Verify examples work

---

*This documentation system provides both quick reference and comprehensive guides for the Hagike toolkit.*
EOF

echo -e "${GREEN}✅ Comprehensive documentation organized${NC}"

# Generate a summary report
echo -e "${BLUE}📊 Documentation Summary:${NC}"
echo -e "  ${GREEN}📁 HTML Documentation:${NC} docs/hagike/index.html"
echo -e "  ${GREEN}📚 Complete API Guide:${NC} docs/manual/API_DOCUMENTATION.md"
echo -e "  ${GREEN}🚀 Quick Start:${NC} docs/manual/QUICK_START_GUIDE.md"
echo -e "  ${GREEN}📋 Function Reference:${NC} docs/reference/FUNCTION_REFERENCE.md"
echo -e "  ${GREEN}🗺️ Navigation Index:${NC} docs/DOCUMENTATION_INDEX.md"

# Check if we can count the documentation
total_functions=$(grep -c "^|.*|.*|.*|$" FUNCTION_REFERENCE.md 2>/dev/null || echo "N/A")
total_examples=$(grep -c "```python" API_DOCUMENTATION.md QUICK_START_GUIDE.md 2>/dev/null || echo "N/A")

echo -e "${YELLOW}📈 Statistics:${NC}"
echo -e "  ${GREEN}Functions documented:${NC} ~${total_functions}"
echo -e "  ${GREEN}Code examples:${NC} ~${total_examples}"
echo -e "  ${GREEN}Documentation files:${NC} 4 comprehensive guides"

echo -e "${GREEN}🎉 Documentation generation complete!${NC}"
echo -e "${BLUE}💡 Next steps:${NC}"
echo -e "  1. Open ${YELLOW}docs/DOCUMENTATION_INDEX.md${NC} to start exploring"
echo -e "  2. For quick start: ${YELLOW}docs/manual/QUICK_START_GUIDE.md${NC}"
echo -e "  3. For HTML docs: ${YELLOW}docs/hagike/index.html${NC}"