package com.server.backend.security;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.server.backend.service.UserService;
import com.server.backend.utilities.JWTContext;

import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.FilterConfig;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;


@Component
public class JWTAuthenticationFilter implements Filter{

    @Autowired
    UserService userServ;
    

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;


         // Skip the filter for these paths
        if ("/user/create".equals(httpRequest.getRequestURI()) || "/user/login".equals(httpRequest.getRequestURI())) {

            chain.doFilter(request, response); 
            return;
        }


        // Retrieve JWT token from header
        String token = httpRequest.getHeader("Authorization");
        if(token == null || token.isEmpty()){
            httpResponse.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Not setted any JWT token");
            return;
        }

        if (token.startsWith("Bearer ")) {
            token = token.substring(7); // removes "Bearer "
        }

        boolean result = userServ.validateToken(token);

        if(result){
            // Token is valid, proceed with the request
            String username = userServ.getUsernameFromToken(token);
            JWTContext.set(username);
            chain.doFilter(request, response);
        } else {
            // Token is invalid
            httpResponse.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid JWT token");
        }
    }

    @Override
    public void destroy() {

        JWTContext.clear();

    }

}
