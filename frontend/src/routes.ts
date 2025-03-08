import {
    type RouteConfig,
    route,
    index,
  } from "@react-router/dev/routes";
  
  export default [
    // parent route
    route("", "./app.tsx", [
    // child routes
        index("./components/Grabacion.tsx"),
        route("Grabacion", "./components/Grabacion.tsx"),
    ]),
  ] satisfies RouteConfig;

