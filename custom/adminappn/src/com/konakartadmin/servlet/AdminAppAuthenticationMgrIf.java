package com.konakartadmin.servlet;

import java.net.URL;

import javax.servlet.http.HttpServletRequest;

/**
 * An implementation of this interface is called from the AdminAppLauncher servlet in order to allow
 * integration with an external authentication service.
 */
public interface AdminAppAuthenticationMgrIf
{
    /**
     * Used to Authenticate a request received by the AdminAppLauncher servlet used to launch the
     * admin app. The HttpServletRequest object must contain at least one parameter:
     * <ul>
     * <li>storeId</li>
     * </ul>
     * The storeId and the userId (returned by this method) are passed by the servlet to the GWT
     * admin application depending on whether or not the authentication is approved.
     * 
     * @param request
     *            The request to authenticate.
     * @return Returns the authenticated user id or null if authentication fails.
     * @throws Exception
     *             If technical problems are encountered during communication with the
     *             authentication system
     */
    public String authenticateRequest(HttpServletRequest request) throws Exception;

    /**
     * If authentication fails, the user may be redirected to an SSO login screen. This method is
     * called to determine whether this is the case since it returns the URL of the login screen.
     * 
     * @param redirectURL
     *            The URL to redirect to (by the SSO system) when the login succeeds. This would
     *            typically be the requestUrl of a request for which the authenticateRequest method
     *            returned null.
     * @return Returns the url of a login screen which the user can use to manually login. This
     *         method may also return null, in this case the user should be redirected to the
     *         standard KonaKartAdmin login screen.
     * @throws Exception
     *             If technical problems are encountered during communication with the
     *             authentication system
     */
    public URL buildManualLoginURL(URL redirectURL) throws Exception;
}
