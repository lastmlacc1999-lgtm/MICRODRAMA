/** @type {import('next').NextConfig} */
// This URL is used by the Next.js server, never by the browser.
const backendUrl = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const nextConfig = {
  // Allow the Arena preview host to load development assets; production is unaffected.
  allowedDevOrigins: ["*.e2b.app"],
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backendUrl}/api/:path*` },
      { source: "/outputs/:path*", destination: `${backendUrl}/outputs/:path*` },
    ];
  },
};
module.exports = nextConfig;
