package com.server.backend.utilities;

public class JWTContext {
    
    private static final ThreadLocal<String> jwtContext = new ThreadLocal<>();

    public static void set(String jwtValidateResponse) {
        jwtContext.set(jwtValidateResponse);
    }

    public static String get() {
        return jwtContext.get();
    }

    public static void clear() {
        jwtContext.remove();
    }
    
}
