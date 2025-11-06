export default [
  {
    files: ['**/*.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: {
          impliedStrict: true,
        },
      },
      globals: {
        // Browser globals
        window: 'readonly',
        document: 'readonly',
        console: 'readonly',
        navigator: 'readonly',
        localStorage: 'readonly',
        alert: 'readonly',
        confirm: 'readonly',
        setTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        clearTimeout: 'readonly',
        requestAnimationFrame: 'readonly',
        fetch: 'readonly',
        performance: 'readonly',
        Blob: 'readonly',
        URL: 'readonly',
        // Optional debugging/tracing globals
        capture: 'readonly',
        // Node.js globals
        process: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        Buffer: 'readonly',
        global: 'readonly',
        // THREE.js (for 3D renderer)
        THREE: 'readonly',
      },
    },
    rules: {
      // Error on actual problems
      'no-undef': 'error',
      'no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      'no-constant-condition': 'warn',
      'no-debugger': 'warn',

      // Relaxed rules - just warnings
      'no-console': 'off', // Console is fine for this project
      semi: ['warn', 'always'],
      quotes: 'off', // Allow both single and double quotes
      indent: 'off', // Don't enforce indent style yet
      'comma-dangle': 'off',
      'no-trailing-spaces': 'off',
    },
  },
  {
    // Ignore patterns
    ignores: [
      'node_modules/**',
      '.build/**',
      'temp/**',
      '__pycache__/**',
      'scripts/GPU_Training/**',
      '*.min.js',
    ],
  },
];
