# Auth0 SSO Test Setup Guide

This guide walks through setting up both SAML and OIDC connections in Auth0's free tier to test the SSO integration.

## 1. Auth0 Setup (General)
1. Sign up for a free Auth0 account at [auth0.com](https://auth0.com).
2. Create a new Auth0 Tenant.
3. You will need users to test with. Go to **User Management > Users** and create a test user (e.g., `sso-test@example.com`).

## 2. Testing SAML 2.0
1. In Auth0, go to **Applications > Applications** and click **Create Application**.
2. Select **Regular Web Application** and name it "TGS SAML App".
3. In the application settings, go to the **Addons** tab and enable **SAML2 Web App**.
4. A settings window will pop up. 
   - **Application Callback URL**: Set this to your ngrok URL + the callback path, e.g., `https://<your-ngrok-url>/auth/saml/<workspace_slug>/callback`.
   - Leave the rest as defaults, scroll to the bottom, and click **Enable**.
5. Go to the **Usage** tab of the SAML2 Web App addon.
   - Download the **Identity Provider Metadata** XML file.
   - Or manually copy the **Issuer** (Entity ID), **Identity Provider Login URL** (SSO URL), and download the **Identity Provider Certificate** (x509 cert).
6. In TGS (using an Admin account in your workspace):
   - Submit a `POST /api/v1/workspace/sso` with `protocol="saml"`.
   - Provide the Entity ID, SSO URL, and the raw text of the x509 certificate.
7. Test the flow by navigating to `https://<your-ngrok-url>/auth/saml/<workspace_slug>/login`. You should be redirected to Auth0, log in, and be redirected back with a valid session cookie.

## 3. Testing OIDC
1. In Auth0, create another **Regular Web Application** and name it "TGS OIDC App".
2. In the application settings:
   - **Allowed Callback URLs**: `https://<your-ngrok-url>/auth/oidc/<workspace_slug>/callback`
   - Save changes.
3. Scroll to the top and grab the **Client ID** and **Client Secret**.
4. To find your **Discovery URL**, go to the bottom of the Settings page and click **Show Advanced Settings** -> **Endpoints**. Copy the **OpenID Configuration** URL.
5. In TGS (using an Admin account in your workspace):
   - Submit a `POST /api/v1/workspace/sso` with `protocol="oidc"`.
   - Provide the Client ID, Client Secret, and Discovery URL.
6. Test the flow by navigating to `https://<your-ngrok-url>/auth/oidc/<workspace_slug>/login`. You should be redirected to Auth0, log in, and be redirected back with a valid session cookie.
