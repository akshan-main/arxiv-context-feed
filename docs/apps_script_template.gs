/**
 * Google Apps Script for creating GitHub Issues from Google Form submissions.
 *
 * This script is triggered when a Google Form is submitted, and it creates
 * a GitHub Issue with the appropriate labels for the config-from-issue workflow.
 *
 * Setup:
 * 1. Create a Google Form with the required fields
 * 2. Open the form's script editor (Extensions > Apps Script)
 * 3. Paste this script
 * 4. Add GITHUB_TOKEN to Script Properties (Project Settings > Script Properties)
 * 5. Set up an "On form submit" trigger
 *
 * Required Script Properties:
 * - GITHUB_TOKEN: Personal access token with 'repo' scope
 * - GITHUB_OWNER: Repository owner (e.g., "your-org")
 * - GITHUB_REPO: Repository name (e.g., "arxiv-context-feed")
 */

// Configuration
const SCRIPT_PROPERTIES = PropertiesService.getScriptProperties();
const GITHUB_TOKEN = SCRIPT_PROPERTIES.getProperty('GITHUB_TOKEN');
const GITHUB_OWNER = SCRIPT_PROPERTIES.getProperty('GITHUB_OWNER') || 'your-org';
const GITHUB_REPO = SCRIPT_PROPERTIES.getProperty('GITHUB_REPO') || 'arxiv-context-feed';

/**
 * Trigger function for form submission.
 * Set this as an "On form submit" trigger in Apps Script.
 *
 * @param {Object} e - Form submit event object
 */
function onFormSubmit(e) {
  try {
    const formResponse = e.response;
    const itemResponses = formResponse.getItemResponses();

    // Extract form data
    const formData = {};
    itemResponses.forEach(function(itemResponse) {
      const title = itemResponse.getItem().getTitle();
      const answer = itemResponse.getResponse();
      formData[title] = answer;
    });

    // Determine change type from form
    const changeType = formData['Change Type'] || 'add';
    const targetType = formData['Target Type'] || 'topic';

    // Build the payload based on target type
    let payload;
    if (targetType === 'topic') {
      payload = buildTopicPayload(formData, changeType);
    } else if (targetType === 'judge') {
      payload = buildJudgePayload(formData, changeType);
    } else {
      throw new Error('Unknown target type: ' + targetType);
    }

    // Create GitHub Issue
    const issueUrl = createGitHubIssue(payload, changeType, targetType);

    // Log success
    Logger.log('Created GitHub Issue: ' + issueUrl);

    // Optionally send confirmation email
    if (formData['Email']) {
      sendConfirmationEmail(formData['Email'], issueUrl, changeType, targetType);
    }

  } catch (error) {
    Logger.log('Error processing form submission: ' + error.toString());
    // Optionally notify admins of errors
  }
}

/**
 * Build payload for topic changes.
 *
 * @param {Object} formData - Form response data
 * @param {string} changeType - 'add', 'edit', or 'remove'
 * @return {Object} Payload object
 */
function buildTopicPayload(formData, changeType) {
  // Parse keywords (comma or newline separated)
  const keywords = parseList(formData['Keywords'] || '');

  // Parse phrases (newline separated for multi-word)
  const phrases = parseList(formData['Phrases'] || '');

  // Parse negatives (comma or newline separated)
  const negatives = parseList(formData['Negatives'] || '');

  // Parse categories (comma separated)
  const categories = parseList(formData['arXiv Categories'] || '');

  const topic = {
    key: normalizeKey(formData['Topic Key'] || formData['Topic Name']),
    name: formData['Topic Name'],
    description: formData['Description'] || '',
    arxiv_categories: categories,
    keywords: keywords,
    phrases: phrases,
    inclusion_notes: formData['Inclusion Notes'] || '',
    exclusion_notes: formData['Exclusion Notes'] || ''
  };

  // Add negatives only if present
  if (negatives.length > 0) {
    topic.negatives = negatives;
  }

  return {
    change_type: changeType,
    target_type: 'topic',
    topic: topic
  };
}

/**
 * Build payload for judge config changes.
 *
 * @param {Object} formData - Form response data
 * @param {string} changeType - 'edit' only for judge
 * @return {Object} Payload object
 */
function buildJudgePayload(formData, changeType) {
  const changes = {};

  // Only include fields that were provided
  if (formData['Strictness']) {
    changes.strictness = formData['Strictness'].toLowerCase();
  }
  if (formData['Model ID']) {
    changes.model_id = formData['Model ID'];
  }
  if (formData['Provider']) {
    changes.provider = formData['Provider'].toLowerCase();
  }

  return {
    change_type: 'edit',
    target_type: 'judge',
    changes: changes
  };
}

/**
 * Create a GitHub Issue with the payload.
 *
 * @param {Object} payload - The JSON payload
 * @param {string} changeType - Change type for title
 * @param {string} targetType - Target type for title
 * @return {string} URL of created issue
 */
function createGitHubIssue(payload, changeType, targetType) {
  const url = 'https://api.github.com/repos/' + GITHUB_OWNER + '/' + GITHUB_REPO + '/issues';

  // Build issue title
  let title;
  if (targetType === 'topic') {
    title = '[Config] ' + capitalize(changeType) + ' topic: ' + payload.topic.name;
  } else {
    title = '[Config] Edit judge settings';
  }

  // Build issue body
  const body = '## Configuration Change Request\n\n' +
               'This issue was automatically created from a Google Form submission.\n\n' +
               '### Payload\n\n' +
               '```json\n' +
               JSON.stringify(payload, null, 2) +
               '\n```\n\n' +
               '---\n' +
               '*Submitted via Google Form at ' + new Date().toISOString() + '*';

  const issueData = {
    title: title,
    body: body,
    labels: ['source:google-form', 'config-change']
  };

  const options = {
    method: 'post',
    headers: {
      'Authorization': 'token ' + GITHUB_TOKEN,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    },
    payload: JSON.stringify(issueData),
    muteHttpExceptions: true
  };

  const response = UrlFetchApp.fetch(url, options);
  const responseCode = response.getResponseCode();

  if (responseCode !== 201) {
    throw new Error('Failed to create issue: ' + response.getContentText());
  }

  const responseData = JSON.parse(response.getContentText());
  return responseData.html_url;
}

/**
 * Send confirmation email to submitter.
 *
 * @param {string} email - Recipient email
 * @param {string} issueUrl - URL of created issue
 * @param {string} changeType - Type of change
 * @param {string} targetType - Target of change
 */
function sendConfirmationEmail(email, issueUrl, changeType, targetType) {
  const subject = 'Config Change Request Submitted';
  const body = 'Thank you for your configuration change request!\n\n' +
               'Change Type: ' + changeType + '\n' +
               'Target: ' + targetType + '\n\n' +
               'A GitHub Issue has been created for review:\n' +
               issueUrl + '\n\n' +
               'You will be notified when the change is approved and merged.';

  GmailApp.sendEmail(email, subject, body);
}

/**
 * Parse a comma or newline separated list.
 *
 * @param {string} input - Input string
 * @return {Array} Array of trimmed, non-empty values
 */
function parseList(input) {
  if (!input) return [];

  return input
    .split(/[,\n]/)
    .map(function(item) { return item.trim(); })
    .filter(function(item) { return item.length > 0; });
}

/**
 * Normalize a topic key from name.
 *
 * @param {string} name - Topic name or key
 * @return {string} Normalized key
 */
function normalizeKey(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Capitalize first letter.
 *
 * @param {string} str - Input string
 * @return {string} Capitalized string
 */
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Test function - creates a test issue.
 * Run this manually to verify setup.
 */
function testCreateIssue() {
  const testPayload = {
    change_type: 'add',
    target_type: 'topic',
    topic: {
      key: 'test-topic',
      name: 'Test Topic',
      description: 'This is a test topic created by Apps Script',
      arxiv_categories: ['cs.AI'],
      keywords: ['test', 'example'],
      phrases: ['test phrase'],
      inclusion_notes: 'Test inclusion notes',
      exclusion_notes: 'Test exclusion notes'
    }
  };

  try {
    const issueUrl = createGitHubIssue(testPayload, 'add', 'topic');
    Logger.log('Test issue created: ' + issueUrl);
  } catch (error) {
    Logger.log('Test failed: ' + error.toString());
  }
}

/**
 * Setup function - verifies configuration.
 * Run this after setting up Script Properties.
 */
function verifySetup() {
  const errors = [];

  if (!GITHUB_TOKEN) {
    errors.push('GITHUB_TOKEN is not set in Script Properties');
  }
  if (!GITHUB_OWNER || GITHUB_OWNER === 'your-org') {
    errors.push('GITHUB_OWNER is not configured');
  }
  if (!GITHUB_REPO || GITHUB_REPO === 'arxiv-context-feed') {
    errors.push('GITHUB_REPO should be configured for your repository');
  }

  if (errors.length > 0) {
    Logger.log('Setup errors:\n- ' + errors.join('\n- '));
    return false;
  }

  Logger.log('Setup verified successfully!');
  Logger.log('Owner: ' + GITHUB_OWNER);
  Logger.log('Repo: ' + GITHUB_REPO);
  Logger.log('Token: ' + GITHUB_TOKEN.substring(0, 4) + '...');
  return true;
}
