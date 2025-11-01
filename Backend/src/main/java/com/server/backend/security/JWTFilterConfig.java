package com.server.backend.security;

import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JWTFilterConfig {
    
    @Bean
    public FilterRegistrationBean<JWTAuthenticationFilter> jwtFilterRegistration(JWTAuthenticationFilter jwtFilter) {
        FilterRegistrationBean<JWTAuthenticationFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(jwtFilter);
        registrationBean.addUrlPatterns("/*");
        return registrationBean;
    }
}
