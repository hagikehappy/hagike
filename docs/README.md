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
