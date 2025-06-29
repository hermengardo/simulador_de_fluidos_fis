const express = require("express");
const app = express();

const PORT = process.env.PORT || 8888;

app.use(express.static('.'));

app.listen(PORT, () => {
    console.log(`Server is running on localhost:${PORT}`);
});

// Assumiremos incompressibilidade do fluido.
// Assumiremos que o fluido é não viscoso.
