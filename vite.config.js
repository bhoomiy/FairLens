import { defineConfig as defineViteConfig } from "vite";
import { defineConfig as defineLovableConfig } from "@lovable.dev/vite-tanstack-config";
import { nitro } from "nitro/vite";

export default defineViteConfig(async (env) => {
  const config = await defineLovableConfig({
    tanstackStart: {
      server: { entry: "server" },
    },

    // Disable Lovable's built-in Nitro target
    nitro: false,
  })(env);

  // For production builds, explicitly build for Vercel
  if (env.command === "build") {
    config.plugins = [
      ...(config.plugins ?? []),
      nitro({
        preset: "vercel",
      }),
    ];
  }

  return config;
});